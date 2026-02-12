"""
Siderust Lab — Interactive Dashboard
=====================================

Launch with::

    streamlit run ui/app.py

Reads structured JSON results from ``results/<date>/<experiment>/<library>.json``
and presents them in an interactive GUI with tabs for Overview, Accuracy,
Performance, Pareto, Trends, Outliers, and Report generation.

Features:
- Dark / light theme toggle (glassmorphism dark theme by default)
- Auto-refresh when new results appear
- Cross-run trend tracking
- Responsive metric cards (no narrow-column overflow)
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import streamlit as st

# ── Ensure the repo root is on sys.path so local imports work ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ui.results_loader import (  # noqa: E402
    ExperimentResult,
    PercentileStats,
    Performance,
    RunInfo,
    WorstCase,
    discover_runs,
    load_results,
)
from ui.theme import inject_theme_css, theme_toggle, is_dark  # noqa: E402

# Tab renderers
from ui.tabs import overview, accuracy, performance, pareto, trends, outliers, reports  # noqa: E402

# ──────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siderust Lab",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Theme CSS
# ──────────────────────────────────────────────────────────
inject_theme_css()

# ──────────────────────────────────────────────────────────
# Auto-refresh (optional)
# ──────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ──────────────────────────────────────────────────────────
# Data loading helpers (cached)
# ──────────────────────────────────────────────────────────
RESULTS_DIR = _REPO_ROOT / "results"


@st.cache_data(ttl=30)
def _discover() -> list[dict]:
    """Discover runs (cached, refresh every 30s)."""
    runs = discover_runs(RESULTS_DIR)
    return [
        {"date": r.date, "experiment": r.experiment, "path": str(r.path), "libraries": r.libraries}
        for r in runs
    ]


@st.cache_data(ttl=30)
def _load(path: str, libs: tuple[str, ...]) -> list[dict]:
    """Load results (cached).  Returns serialisable dicts."""
    run = RunInfo(date="", experiment="", path=Path(path))
    results = load_results(run, list(libs) if libs else None)
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
    """Re-hydrate ExperimentResult objects from cached dicts, using safe from_dict()."""
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
            angular_error_mas=PercentileStats.from_dict(d.get("angular_error_mas")),
            closure_error_rad=PercentileStats.from_dict(d.get("closure_error_rad")),
            matrix_frobenius=PercentileStats.from_dict(d.get("matrix_frobenius")),
            nan_count=d.get("nan_count", 0),
            inf_count=d.get("inf_count", 0),
            worst_cases=[WorstCase.from_dict(w, i) for i, w in enumerate(d.get("worst_cases", []))],
            gmst_error_rad=PercentileStats.from_dict(d.get("gmst_error_rad")),
            gmst_error_arcsec=PercentileStats.from_dict(d.get("gmst_error_arcsec")),
            era_error_rad=PercentileStats.from_dict(d.get("era_error_rad")),
            performance=Performance.from_dict(d.get("performance")),
            reference_performance=Performance.from_dict(d.get("reference_performance")),
            run_metadata=d.get("run_metadata", {}),
            source_path=d.get("source_path", ""),
            run_date=d.get("run_date", ""),
        )
        results.append(r)
    return results


def _selection_hash(date: str, experiment: str, libs: list[str]) -> str:
    """Hash the current sidebar selection to detect changes."""
    return hashlib.md5(f"{date}|{experiment}|{sorted(libs)}".encode()).hexdigest()


# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔭 Siderust Lab")
    st.caption("Benchmark & validation explorer")

    st.divider()
    theme_toggle()

    # Auto-refresh toggle
    if _HAS_AUTOREFRESH:
        auto_refresh = st.toggle("🔄 Auto-refresh (30s)", value=False, key="auto_refresh")
        if auto_refresh:
            st_autorefresh(interval=30_000, limit=None, key="autorefresh_timer")
    else:
        st.caption("Install `streamlit-autorefresh` for live reload.")

    st.divider()

    runs_raw = _discover()
    if not runs_raw:
        st.warning(
            "No results found.  Run the pipeline first:\n"
            "```bash\nbash run.sh\n```"
        )
        st.stop()

    # Group by date
    dates = sorted({r["date"] for r in runs_raw}, reverse=True)
    selected_date = st.selectbox("📅 Run date", dates)

    # Filter experiments for this date
    experiments_list = sorted({r["experiment"] for r in runs_raw if r["date"] == selected_date})
    selected_experiment = st.selectbox("🧪 Experiment", experiments_list)

    # Find matching run
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

# ──────────────────────────────────────────────────────────
# Clear stale report when selection changes
# ──────────────────────────────────────────────────────────
sel_hash = _selection_hash(selected_date, selected_experiment, selected_libs)
if st.session_state.get("_sel_hash") != sel_hash:
    st.session_state["_sel_hash"] = sel_hash
    # Clear stale report content
    for key in ("report_content", "report_ext", "report_mime"):
        st.session_state.pop(key, None)

# ──────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────
cached_dicts = _load(run_info["path"], tuple(selected_libs))
results: list[ExperimentResult] = _reconstruct(cached_dicts)

if not results:
    st.error("Could not load any results for this selection.")
    st.stop()

# ──────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────
tab_labels = [
    "📊 Overview",
    "🎯 Accuracy",
    "⚡ Performance",
    "🏔 Pareto",
    "📈 Trends",
    "🔍 Outliers",
    "📄 Reports",
]
(
    tab_overview,
    tab_accuracy,
    tab_performance,
    tab_pareto,
    tab_trends,
    tab_outliers,
    tab_reports,
) = st.tabs(tab_labels)

with tab_overview:
    st.header(f"{selected_experiment}")
    overview.render(results, selected_experiment)

with tab_accuracy:
    st.header("Accuracy Analysis")
    accuracy.render(results, selected_experiment)

with tab_performance:
    st.header("Performance Analysis")
    performance.render(results, selected_experiment)

with tab_pareto:
    st.header("Pareto — Accuracy vs Latency")
    pareto.render(results, selected_experiment)

with tab_trends:
    st.header("📈 Trends — Cross-Run Comparison")
    trends.render(results, selected_experiment, RESULTS_DIR)

with tab_outliers:
    st.header("Worst-N Outliers")
    outliers.render(results, selected_experiment)

with tab_reports:
    st.header("Generate Report")
    reports.render(results, selected_experiment, selected_date, _REPO_ROOT)

# ──────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Siderust Lab — "
    f"Loaded {len(results)} result(s) for **{selected_experiment}** ({selected_date}).  "
    "Reference: ERFA/SOFA-derived outputs."
)
