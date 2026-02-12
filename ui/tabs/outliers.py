"""Outliers tab — worst-N cases table + scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.report_generator import color, fmt
from ui.theme import is_dark, plotly_layout

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def render(results: list["ExperimentResult"], experiment: str) -> None:
    dark = is_dark()

    st.markdown(
        "Cases with the largest error compared to the reference.  "
        "Use this table to investigate whether outliers cluster at "
        "specific epochs or sky positions."
    )

    any_outliers = False
    for r in results:
        if not r.worst_cases:
            continue
        any_outliers = True

        st.markdown(
            f'<div class="section-header">'
            f'{r.candidate_library} vs {r.reference_library}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Use dynamic column name based on primary error label
        error_label = r.primary_error_label

        wc_rows = []
        for i, wc in enumerate(r.worst_cases):
            approx_year = 2000.0 + ((wc.jd_tt or 2451545.0) - 2451545.0) / 365.25
            wc_rows.append({
                "#": i + 1,
                "Case ID": wc.case_id,
                "JD(TT)": f"{wc.jd_tt:.6f}" if wc.jd_tt else "—",
                "≈ Year": f"{approx_year:.1f}",
                error_label: fmt(wc.angular_error_mas, 3),
                "Notes": wc.notes or "",
            })

        if wc_rows:
            df_wc = pd.DataFrame(wc_rows)
            st.dataframe(df_wc, width="stretch", hide_index=True)

            # Scatter: error vs epoch
            fig_outlier = go.Figure()
            jds = [wc.jd_tt for wc in r.worst_cases if wc.jd_tt]
            errs = [wc.angular_error_mas for wc in r.worst_cases if wc.jd_tt]
            fig_outlier.add_trace(
                go.Scatter(
                    x=jds, y=errs,
                    mode="markers",
                    marker=dict(size=10, color=color(r.candidate_library),
                                line=dict(width=1, color="rgba(255,255,255,0.3)")),
                    hovertemplate="JD: %{x:.2f}<br>Error: %{y:.3f}<extra></extra>",
                )
            )
            fig_outlier.update_layout(
                title=f"Outlier Error vs Epoch — {r.candidate_library}",
                xaxis_title="Julian Date (TT)",
                yaxis_title=error_label,
                **plotly_layout(),
            )
            st.plotly_chart(fig_outlier, width="stretch", key=f"outlier_{r.candidate_library}")

            csv_wc = df_wc.to_csv(index=False)
            st.download_button(
                f"⬇ Download outliers — {r.candidate_library} (CSV)",
                data=csv_wc,
                file_name=f"{experiment}_{r.candidate_library}_outliers.csv",
                mime="text/csv",
                key=f"dl_outlier_{r.candidate_library}",
            )

    if not any_outliers:
        # Show fallback summary when no worst_cases recorded
        st.info("No worst-case outlier data available for this experiment.")
        _fallback_summary(results)


def _fallback_summary(results: list["ExperimentResult"]) -> None:
    """Show p99/max as a summary when no explicit worst cases exist."""
    rows = []
    for r in results:
        s = r.primary_error_stats
        if not s.is_empty():
            rows.append({
                "Library": r.candidate_library,
                f"p99 {r.primary_error_label}": fmt(s.p99, 3),
                f"max {r.primary_error_label}": fmt(s.max, 3),
            })
    if rows:
        st.markdown("**Tail-end summary (p99 / max):**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
