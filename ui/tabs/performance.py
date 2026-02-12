"""Performance tab — latency, throughput, detailed table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from ui.report_generator import make_latency_bar, make_throughput_bar
from ui.theme import is_dark

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def render(results: list["ExperimentResult"], experiment: str) -> None:
    dark = is_dark()
    perf_results = [r for r in results if r.has_performance]

    if not perf_results:
        st.info("No performance data recorded for this experiment.")
        return

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_lat = make_latency_bar(perf_results, dark=dark)
        st.plotly_chart(fig_lat, width="stretch", key="perf_lat")
    with pcol2:
        fig_thr = make_throughput_bar(perf_results, dark=dark)
        st.plotly_chart(fig_thr, width="stretch", key="perf_thr")

    # Detailed table
    st.markdown('<div class="section-header">Detailed Performance</div>', unsafe_allow_html=True)
    perf_rows = []
    for r in perf_results:
        p = r.performance
        rp = r.reference_performance
        perf_rows.append({
            "Library": r.candidate_library,
            "Latency (ns/op)": f"{p.per_op_ns:,.1f}" if p.per_op_ns else "—",
            "Latency (µs/op)": f"{p.per_op_us:,.2f}" if p.per_op_us else "—",
            "Throughput (ops/s)": f"{p.throughput_ops_s:,.0f}" if p.throughput_ops_s else "—",
            "Batch size": str(p.batch_size) if p.batch_size else "—",
            "Total time (ms)": f"{p.total_ns / 1e6:,.1f}" if p.total_ns else "—",
            "Ref latency (ns)": f"{rp.per_op_ns:,.1f}" if rp.per_op_ns else "—",
            "Speedup": f"{r.speedup_vs_ref:.1f}×" if r.speedup_vs_ref else "—",
        })
    st.dataframe(pd.DataFrame(perf_rows), width="stretch", hide_index=True)

    # Definitions
    with st.expander("📖 Metric definitions"):
        st.markdown("""
**Latency (ns/op):** Time per single transformation call.  Lower is better.

**Throughput (ops/s):** Transformations completed per second.  Higher is better.

**Speedup:** Ratio of reference latency to candidate latency.

*Note:* Timing includes the transformation computation only — not I/O or serialization.
""")

    # Export
    if perf_rows:
        df_perf = pd.DataFrame(perf_rows)
        st.download_button(
            "⬇ Download performance table (CSV)",
            data=df_perf.to_csv(index=False),
            file_name=f"{experiment}_performance.csv",
            mime="text/csv",
        )
