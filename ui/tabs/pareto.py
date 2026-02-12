"""Pareto tab — accuracy vs latency scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from ui.report_generator import fmt, make_pareto_scatter
from ui.theme import is_dark

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def render(results: list["ExperimentResult"], experiment: str) -> None:
    dark = is_dark()

    st.markdown(
        "Each dot represents one library.  Libraries closer to the **bottom-left** "
        "corner are on the Pareto frontier (best trade-off between accuracy and speed)."
    )

    pareto_results = [r for r in results if r.has_performance and r.has_accuracy]
    if len(pareto_results) < 1:
        st.info("Need at least one library with both accuracy and performance data.")
        return

    fig_pareto = make_pareto_scatter(pareto_results, dark=dark)
    st.plotly_chart(fig_pareto, width="stretch", key="pareto")

    # Reference table
    pareto_rows = []
    for r in pareto_results:
        s = r.primary_error_stats
        pareto_rows.append({
            "Library": r.candidate_library,
            f"p99 {r.primary_error_label}": fmt(s.p99, 3),
            "Latency (ns)": f"{r.performance.per_op_ns:,.0f}" if r.performance.per_op_ns else "—",
            "Speedup": f"{r.speedup_vs_ref:.1f}×" if r.speedup_vs_ref else "—",
        })
    st.dataframe(pd.DataFrame(pareto_rows), width="stretch", hide_index=True)

    with st.expander("📖 How to read this plot"):
        st.markdown("""
- **X-axis:** Latency per operation (ns), log scale.  Further left = faster.
- **Y-axis:** 99th-percentile accuracy error.  Further down = more accurate.
- The ideal library lives in the **bottom-left corner** (fast *and* accurate).
- Points on the **Pareto frontier** are libraries where you can't improve one metric without sacrificing the other.
""")
