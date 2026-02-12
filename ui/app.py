"""
Siderust Lab — Interactive Dashboard
=====================================

Launch with::

    streamlit run ui/app.py

Reads structured JSON results from ``results/<date>/<experiment>/<library>.json``
and presents them in an interactive GUI with tabs for Overview, Accuracy,
Performance, Pareto, Outliers, and Report generation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Ensure the repo root is on sys.path so local imports work ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ui.results_loader import (  # noqa: E402
    ExperimentResult,
    RunInfo,
    discover_runs,
    load_results,
)
from ui.report_generator import (  # noqa: E402
    PLOTLY_LAYOUT_DEFAULTS,
    LIB_COLORS,
    _color,
    _fmt,
    make_accuracy_bar,
    make_accuracy_box,
    make_latency_bar,
    make_throughput_bar,
    make_pareto_scatter,
    make_gmst_era_bar,
    one_line_summary,
    generate_html_report,
    generate_markdown_report,
)

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siderust Lab",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten spacing */
    .block-container { padding-top: 1.5rem; }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    /* Tag badges */
    .lib-tag {
        display: inline-block; background: #e0f2fe; color: #0369a1;
        border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.8rem;
        margin-right: 0.25rem;
    }
    .ref-tag {
        display: inline-block; background: #dcfce7; color: #166534;
        border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# Sidebar – data discovery & selection
# ──────────────────────────────────────────────────────────────────
RESULTS_DIR = _REPO_ROOT / "results"


@st.cache_data(ttl=30)
def _discover() -> list[dict]:
    """Discover runs (cached, refresh every 30s)."""
    runs = discover_runs(RESULTS_DIR)
    return [
        {"date": r.date, "experiment": r.experiment, "path": str(r.path), "libraries": r.libraries}
        for r in runs
    ]


@st.cache_data(ttl=60)
def _load(path: str, libs: tuple[str, ...]) -> list[dict]:
    """Load results (cached).  Returns serialisable dicts to keep Streamlit happy."""
    run = RunInfo(date="", experiment="", path=Path(path))
    results = load_results(run, list(libs) if libs else None)
    # Re-serialise through JSON round-trip so Streamlit's cache can hash them
    import dataclasses
    out = []
    for r in results:
        d = {}
        for f in dataclasses.fields(r):
            v = getattr(r, f.name)
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                d[f.name] = dataclasses.asdict(v)
            elif isinstance(v, list) and v and dataclasses.is_dataclass(v[0]):
                d[f.name] = [dataclasses.asdict(x) for x in v]
            else:
                d[f.name] = v
        out.append(d)
    return out


def _reconstruct(dicts: list[dict]) -> list[ExperimentResult]:
    """Re-hydrate ExperimentResult objects from cached dicts."""
    from ui.results_loader import PercentileStats, WorstCase, Performance

    results = []
    for d in dicts:
        r = ExperimentResult(
            experiment=d.get("experiment", ""),
            candidate_library=d.get("candidate_library", ""),
            reference_library=d.get("reference_library", ""),
            mode=d.get("mode", ""),
            alignment=d.get("alignment", {}),
            input_count=d.get("input_count", 0),
            seed=d.get("seed"),
            epoch_range=d.get("epoch_range", []),
            angular_error_mas=PercentileStats(**d["angular_error_mas"]) if "angular_error_mas" in d else PercentileStats(),
            closure_error_rad=PercentileStats(**d["closure_error_rad"]) if "closure_error_rad" in d else PercentileStats(),
            matrix_frobenius=PercentileStats(**d["matrix_frobenius"]) if "matrix_frobenius" in d else PercentileStats(),
            nan_count=d.get("nan_count", 0),
            inf_count=d.get("inf_count", 0),
            worst_cases=[WorstCase(**w) for w in d.get("worst_cases", [])],
            gmst_error_rad=PercentileStats(**d["gmst_error_rad"]) if "gmst_error_rad" in d else PercentileStats(),
            gmst_error_arcsec=PercentileStats(**d["gmst_error_arcsec"]) if "gmst_error_arcsec" in d else PercentileStats(),
            era_error_rad=PercentileStats(**d["era_error_rad"]) if "era_error_rad" in d else PercentileStats(),
            performance=Performance(**d["performance"]) if "performance" in d else Performance(),
            reference_performance=Performance(**d["reference_performance"]) if "reference_performance" in d else Performance(),
            run_metadata=d.get("run_metadata", {}),
            source_path=d.get("source_path", ""),
            run_date=d.get("run_date", ""),
        )
        results.append(r)
    return results


# ── Build sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/telescope.png", width=40)
    st.title("Siderust Lab")
    st.caption("Benchmark & validation explorer")

    runs_raw = _discover()
    if not runs_raw:
        st.warning("No results found.  Run the pipeline first:\n```bash\nbash run.sh\n```")
        st.stop()

    # Group by date
    dates = sorted({r["date"] for r in runs_raw}, reverse=True)
    selected_date = st.selectbox("📅 Run date", dates)

    # Filter experiments for this date
    experiments = sorted({r["experiment"] for r in runs_raw if r["date"] == selected_date})
    selected_experiment = st.selectbox("🧪 Experiment", experiments)

    # Find the matching run info
    matching = [r for r in runs_raw if r["date"] == selected_date and r["experiment"] == selected_experiment]
    if not matching:
        st.error("No matching run found.")
        st.stop()
    run_info = matching[0]

    # Library selection
    available_libs = run_info["libraries"]
    selected_libs = st.multiselect(
        "📚 Libraries to compare",
        available_libs,
        default=available_libs,
    )
    if not selected_libs:
        st.info("Select at least one library.")
        st.stop()

    st.divider()
    st.caption(f"Results path: `{run_info['path']}`")

# ── Load data ──
cached_dicts = _load(run_info["path"], tuple(selected_libs))
results: list[ExperimentResult] = _reconstruct(cached_dicts)

if not results:
    st.error("Could not load any results for this selection.")
    st.stop()


# ──────────────────────────────────────────────────────────────────
# Experiment descriptions
# ──────────────────────────────────────────────────────────────────
EXPERIMENT_DESCRIPTIONS = {
    "frame_rotation_bpn": (
        "**Bias-Precession-Nutation (BPN) direction transform** — rotates ICRS "
        "direction vectors to the True-of-Date frame using the full BPN rotation "
        "matrix.  The reference is ERFA's `eraPnm06a` (IAU 2006/2000A)."
    ),
    "gmst_era": (
        "**Greenwich Mean Sidereal Time (GMST) & Earth Rotation Angle (ERA)** — "
        "computes GMST and ERA for a range of Julian Dates.  "
        "The reference is ERFA's `eraGmst06` / `eraEra00`."
    ),
}


# ──────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────
tab_overview, tab_accuracy, tab_performance, tab_pareto, tab_outliers, tab_reports = st.tabs(
    ["📊 Overview", "🎯 Accuracy", "⚡ Performance", "🏔 Pareto", "🔍 Outliers", "📄 Reports"]
)


# ════════════════════════════════════════════════════════════════════
# TAB: Overview
# ════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header(f"{selected_experiment}")
    desc = EXPERIMENT_DESCRIPTIONS.get(selected_experiment, "")
    if desc:
        st.markdown(desc)

    # One-line summary
    st.info(one_line_summary(results))

    # Key metrics in columns
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            st.markdown(f'<span class="lib-tag">{r.candidate_library}</span> vs <span class="ref-tag">{r.reference_library}</span>', unsafe_allow_html=True)
            s = r.primary_error_stats
            label = r.primary_error_label

            if s.p50 is not None:
                st.metric(f"{label} — p50", _fmt(s.p50, 3))
            if s.p99 is not None:
                st.metric(f"{label} — p99", _fmt(s.p99, 3))
            if s.max is not None:
                st.metric(f"{label} — max", _fmt(s.max, 3))
            if r.performance.per_op_ns is not None:
                st.metric("Latency (ns/op)", f"{r.performance.per_op_ns:,.0f}")
            if r.speedup_vs_ref is not None:
                st.metric("Speedup vs reference", f"{r.speedup_vs_ref:.1f}×")
            st.metric("NaN / Inf", f"{r.nan_count} / {r.inf_count}")

    # Alignment checklist
    st.subheader("Alignment Checklist")
    st.markdown(
        "These are the assumptions locked for this run.  "
        "Differences in models/settings across libraries explain "
        "accuracy gaps — not implementation bugs."
    )
    alignment = results[0].alignment
    mode = alignment.get("mode", "—")
    note = alignment.get("note", "")
    st.markdown(f"**Mode:** `{mode}`")
    if note:
        st.caption(note)

    with st.expander("Full alignment details", expanded=False):
        st.json(alignment)

    # Run metadata
    st.subheader("Run Metadata")
    meta = results[0].run_metadata
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.markdown(f"**Date:** {meta.get('date', '—')}")
    with meta_cols[1]:
        st.markdown(f"**CPU:** {meta.get('cpu', '—')}")
    with meta_cols[2]:
        st.markdown(f"**OS:** {meta.get('os', '—')}")
    with meta_cols[3]:
        toolchain = meta.get("toolchain", {})
        parts = [f"{k}: {v}" for k, v in toolchain.items()]
        st.markdown(f"**Toolchain:** {', '.join(parts)}")

    git_shas = meta.get("git_shas", {})
    if git_shas:
        sha_parts = [f"`{name}` = `{sha}`" for name, sha in git_shas.items()]
        st.markdown(f"**Git SHAs:** {' · '.join(sha_parts)}")

    st.markdown(f"**Input count:** {results[0].input_count}  ·  **Seed:** {results[0].seed}")


# ════════════════════════════════════════════════════════════════════
# TAB: Accuracy
# ════════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.header("Accuracy Analysis")

    if selected_experiment == "frame_rotation_bpn":
        # Bar chart: percentile summaries
        fig_bar = make_accuracy_bar(results, title="Angular Error — Percentile Summary")
        st.plotly_chart(fig_bar, use_container_width=True, key="acc_bar")

        # Box-like summary
        fig_box = make_accuracy_box(results)
        st.plotly_chart(fig_box, use_container_width=True, key="acc_box")

        # ECDF approximation using available percentiles
        st.subheader("Approximate ECDF")
        fig_ecdf = go.Figure()
        for r in results:
            s = r.angular_error_mas
            # Build pseudo-CDF points from known percentiles
            points = []
            if s.min is not None:
                points.append((s.min, 0.0))
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
                fig_ecdf.add_trace(go.Scatter(
                    x=list(xs), y=list(ys),
                    mode="lines+markers",
                    name=r.candidate_library,
                    line=dict(color=_color(r.candidate_library)),
                ))

        fig_ecdf.update_layout(
            title="Approximate ECDF — Angular Error (mas)",
            xaxis_title="Angular error (mas)",
            yaxis_title="Cumulative fraction",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_ecdf, use_container_width=True, key="acc_ecdf")

        # Closure error
        st.subheader("Closure Error (round-trip A→B→A)")
        closure_data = []
        for r in results:
            s = r.closure_error_rad
            if not s.is_empty():
                closure_data.append({
                    "Library": r.candidate_library,
                    "p50 (rad)": _fmt(s.p50, 12),
                    "p99 (rad)": _fmt(s.p99, 12),
                    "max (rad)": _fmt(s.max, 12),
                    "RMS (rad)": _fmt(s.rms, 12),
                })
        if closure_data:
            st.dataframe(pd.DataFrame(closure_data), use_container_width=True, hide_index=True)

        # Matrix Frobenius norm
        st.subheader("Matrix Frobenius Norm (BPN matrix difference)")
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
            st.dataframe(pd.DataFrame(frob_data), use_container_width=True, hide_index=True)

    elif selected_experiment == "gmst_era":
        fig_gmst = make_gmst_era_bar(results)
        st.plotly_chart(fig_gmst, use_container_width=True, key="gmst_bar")

        # Detailed table
        st.subheader("Detailed GMST / ERA Error")
        detail_data = []
        for r in results:
            gs = r.gmst_error_arcsec
            es = r.era_error_rad
            detail_data.append({
                "Library": r.candidate_library,
                "GMST p50 (arcsec)": _fmt(gs.p50, 6),
                "GMST p99 (arcsec)": _fmt(gs.p99, 6),
                "GMST max (arcsec)": _fmt(gs.max, 6),
                "ERA p50 (rad)": f"{es.p50:.2e}" if es.p50 is not None else "—",
                "ERA max (rad)": f"{es.max:.2e}" if es.max is not None else "—",
            })
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

    else:
        st.info(f"No specialised accuracy view for experiment `{selected_experiment}` yet.")

    # CSV export
    st.subheader("Export")
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
            file_name=f"{selected_experiment}_accuracy.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════
# TAB: Performance
# ════════════════════════════════════════════════════════════════════
with tab_performance:
    st.header("Performance Analysis")

    perf_results = [r for r in results if r.has_performance]

    if not perf_results:
        st.info("No performance data recorded for this experiment.")
    else:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig_lat = make_latency_bar(perf_results)
            st.plotly_chart(fig_lat, use_container_width=True, key="perf_lat")
        with pcol2:
            fig_thr = make_throughput_bar(perf_results)
            st.plotly_chart(fig_thr, use_container_width=True, key="perf_thr")

        # Detailed performance table
        st.subheader("Detailed Performance")
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
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

        # Key definitions
        with st.expander("📖 Metric definitions"):
            st.markdown("""
**Latency (ns/op):** Time per single transformation call, in nanoseconds.
Lower is better.

**Throughput (ops/s):** Number of transformations completed per second.
Higher is better.

**Speedup:** Ratio of reference latency to candidate latency.  A value
of 10× means the candidate is 10 times faster than the reference.

*Note:* Timing includes the transformation computation only — not I/O
or JSON serialization.
""")

        # Export
        if perf_rows:
            df_perf = pd.DataFrame(perf_rows)
            st.download_button(
                "⬇ Download performance table (CSV)",
                data=df_perf.to_csv(index=False),
                file_name=f"{selected_experiment}_performance.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════
# TAB: Pareto
# ════════════════════════════════════════════════════════════════════
with tab_pareto:
    st.header("Pareto — Accuracy vs Latency")
    st.markdown(
        "Each dot represents one library.  Libraries closer to the **bottom-left** "
        "corner are on the Pareto frontier (best trade-off between accuracy and speed)."
    )

    pareto_results = [r for r in results if r.has_performance and r.has_accuracy]
    if len(pareto_results) < 1:
        st.info("Need at least one library with both accuracy and performance data.")
    else:
        fig_pareto = make_pareto_scatter(pareto_results)
        st.plotly_chart(fig_pareto, use_container_width=True, key="pareto")

        # Also show a quick reference table
        pareto_rows = []
        for r in pareto_results:
            s = r.primary_error_stats
            pareto_rows.append({
                "Library": r.candidate_library,
                f"p99 {r.primary_error_label}": _fmt(s.p99, 3),
                "p50 latency (ns)": f"{r.performance.per_op_ns:,.0f}" if r.performance.per_op_ns else "—",
                "Speedup": f"{r.speedup_vs_ref:.1f}×" if r.speedup_vs_ref else "—",
            })
        st.dataframe(pd.DataFrame(pareto_rows), use_container_width=True, hide_index=True)

        with st.expander("📖 How to read this plot"):
            st.markdown("""
- **X-axis:** Median latency per operation (nanoseconds), log scale.
  Further left = faster.
- **Y-axis:** 99th-percentile accuracy error.
  Further down = more accurate.
- The ideal library lives in the **bottom-left corner** (fast *and* accurate).
- Points on the **Pareto frontier** (the lower-left envelope) represent libraries
  where you can't improve one metric without sacrificing the other.
""")


# ════════════════════════════════════════════════════════════════════
# TAB: Outliers
# ════════════════════════════════════════════════════════════════════
with tab_outliers:
    st.header("Worst-N Outliers")
    st.markdown(
        "Cases with the largest error compared to the reference.  "
        "Use this table to investigate whether outliers cluster at "
        "specific epochs or sky positions."
    )

    for r in results:
        if not r.worst_cases:
            continue

        st.subheader(f"{r.candidate_library} vs {r.reference_library}")

        wc_rows = []
        for i, wc in enumerate(r.worst_cases):
            # Convert JD to approximate calendar date for context
            approx_year = 2000.0 + ((wc.jd_tt or 2451545.0) - 2451545.0) / 365.25
            wc_rows.append({
                "#": i + 1,
                "Case ID": wc.case_id,
                "JD(TT)": f"{wc.jd_tt:.6f}" if wc.jd_tt else "—",
                "≈ Year": f"{approx_year:.1f}",
                "Error (mas)": _fmt(wc.angular_error_mas, 3),
                "Notes": wc.notes or "",
            })

        if wc_rows:
            df_wc = pd.DataFrame(wc_rows)
            st.dataframe(df_wc, use_container_width=True, hide_index=True)

            # Scatter: error vs epoch
            fig_outlier = go.Figure()
            jds = [wc.jd_tt for wc in r.worst_cases if wc.jd_tt]
            errs = [wc.angular_error_mas for wc in r.worst_cases if wc.jd_tt]
            fig_outlier.add_trace(go.Scatter(
                x=jds, y=errs,
                mode="markers",
                marker=dict(size=10, color=_color(r.candidate_library)),
                hovertemplate="JD: %{x:.2f}<br>Error: %{y:.3f} mas<extra></extra>",
            ))
            fig_outlier.update_layout(
                title=f"Outlier Error vs Epoch — {r.candidate_library}",
                xaxis_title="Julian Date (TT)",
                yaxis_title="Angular error (mas)",
                **PLOTLY_LAYOUT_DEFAULTS,
            )
            st.plotly_chart(fig_outlier, use_container_width=True, key=f"outlier_{r.candidate_library}")

            # Export
            csv_wc = df_wc.to_csv(index=False)
            st.download_button(
                f"⬇ Download outliers — {r.candidate_library} (CSV)",
                data=csv_wc,
                file_name=f"{selected_experiment}_{r.candidate_library}_outliers.csv",
                mime="text/csv",
                key=f"dl_outlier_{r.candidate_library}",
            )

    # Check if any results had outliers
    if not any(r.worst_cases for r in results):
        st.info("No worst-case outlier data available for this experiment.")


# ════════════════════════════════════════════════════════════════════
# TAB: Reports
# ════════════════════════════════════════════════════════════════════
with tab_reports:
    st.header("Generate Report")
    st.markdown(
        "Create a shareable HTML or Markdown report for the current selection.  "
        "Reports include all metrics, plots, the alignment checklist, and reproduction commands."
    )

    report_format = st.radio(
        "Report format",
        ["HTML (with embedded plots)", "Markdown (text only)"],
        horizontal=True,
    )

    col_gen, col_save = st.columns([1, 1])

    with col_gen:
        if st.button("🚀 Generate report", type="primary", use_container_width=True):
            with st.spinner("Generating report…"):
                if "HTML" in report_format:
                    try:
                        report_content = generate_html_report(results)
                        st.session_state["report_content"] = report_content
                        st.session_state["report_ext"] = "html"
                        st.session_state["report_mime"] = "text/html"
                    except Exception as e:
                        st.error(f"Error generating HTML report: {e}")
                        st.info("Falling back to Markdown (install `kaleido` for PNG plot embedding).")
                        report_content = generate_markdown_report(results)
                        st.session_state["report_content"] = report_content
                        st.session_state["report_ext"] = "md"
                        st.session_state["report_mime"] = "text/markdown"
                else:
                    report_content = generate_markdown_report(results)
                    st.session_state["report_content"] = report_content
                    st.session_state["report_ext"] = "md"
                    st.session_state["report_mime"] = "text/markdown"

    with col_save:
        if st.button("💾 Save to reports/ directory", use_container_width=True):
            reports_dir = _REPO_ROOT / "reports" / selected_date / selected_experiment
            reports_dir.mkdir(parents=True, exist_ok=True)

            if "HTML" in report_format:
                out_path = reports_dir / "index.html"
                try:
                    generate_html_report(results, output_path=out_path)
                    st.success(f"✓ Saved to `{out_path.relative_to(_REPO_ROOT)}`")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                out_path = reports_dir / "report.md"
                out_path.write_text(generate_markdown_report(results))
                st.success(f"✓ Saved to `{out_path.relative_to(_REPO_ROOT)}`")

    # Show preview + download
    if "report_content" in st.session_state:
        ext = st.session_state["report_ext"]
        content = st.session_state["report_content"]
        mime = st.session_state["report_mime"]

        st.divider()
        st.download_button(
            f"⬇ Download report (.{ext})",
            data=content,
            file_name=f"{selected_experiment}_report.{ext}",
            mime=mime,
            key="dl_report",
        )

        if ext == "html":
            st.subheader("Preview")
            st.components.v1.html(content, height=800, scrolling=True)
        else:
            st.subheader("Preview")
            st.markdown(content)


# ──────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Siderust Lab — "
    f"Loaded {len(results)} result(s) for **{selected_experiment}** ({selected_date}).  "
    "Reference: ERFA/SOFA-derived outputs."
)
