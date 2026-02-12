"""
Theme system — dark / light toggle with glassmorphism cards.
============================================================

Provides:
- ``inject_theme_css()`` — call once to inject the active CSS.
- ``theme_toggle()``    — sidebar widget that flips the theme.
- ``is_dark()``         — ``True`` when the dark theme is active.
- ``plotly_template()`` — ``"plotly_dark"`` or ``"plotly_white"``.
"""

from __future__ import annotations

import streamlit as st

# ── Colour tokens ────────────────────────────────────────

DARK = {
    "bg":          "#0f172a",
    "surface":     "rgba(30, 41, 59, 0.80)",
    "card":        "rgba(30, 41, 59, 0.65)",
    "border":      "rgba(255, 255, 255, 0.08)",
    "text":        "#e2e8f0",
    "text_muted":  "#94a3b8",
    "accent":      "#3b82f6",
    "accent2":     "#06b6d4",
    "success":     "#22c55e",
    "warning":     "#f59e0b",
    "error":       "#ef4444",
    "lib_badge_bg":  "rgba(59, 130, 246, 0.18)",
    "lib_badge_fg":  "#93c5fd",
    "ref_badge_bg":  "rgba(34, 197, 94, 0.18)",
    "ref_badge_fg":  "#86efac",
    "metric_bg":     "rgba(30, 41, 59, 0.60)",
    "metric_border": "rgba(255, 255, 255, 0.06)",
    "tab_bg":        "rgba(30, 41, 59, 0.40)",
    "input_bg":      "rgba(15, 23, 42, 0.60)",
}

LIGHT = {
    "bg":          "#f8fafc",
    "surface":     "rgba(255, 255, 255, 0.90)",
    "card":        "rgba(255, 255, 255, 0.85)",
    "border":      "#e2e8f0",
    "text":        "#0f172a",
    "text_muted":  "#64748b",
    "accent":      "#2563eb",
    "accent2":     "#0891b2",
    "success":     "#16a34a",
    "warning":     "#d97706",
    "error":       "#dc2626",
    "lib_badge_bg":  "#dbeafe",
    "lib_badge_fg":  "#1d4ed8",
    "ref_badge_bg":  "#dcfce7",
    "ref_badge_fg":  "#166534",
    "metric_bg":     "#ffffff",
    "metric_border": "#e2e8f0",
    "tab_bg":        "#f1f5f9",
    "input_bg":      "#ffffff",
}


# ── Public helpers ───────────────────────────────────────

def is_dark() -> bool:
    return st.session_state.get("theme", "dark") == "dark"


def tokens() -> dict[str, str]:
    return DARK if is_dark() else LIGHT


def plotly_template() -> str:
    return "plotly_dark" if is_dark() else "plotly_white"


def plotly_layout(*, extra: dict | None = None) -> dict:
    """Return the full Plotly layout defaults for the active theme."""
    t = tokens()
    base = dict(
        template=plotly_template(),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=t["text"]),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)" if is_dark() else "#ffffff",
    )
    if extra:
        base.update(extra)
    return base


# ── Sidebar toggle ───────────────────────────────────────

def theme_toggle() -> None:
    """Render a theme toggle in the sidebar."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    label = "☀️ Light mode" if is_dark() else "🌙 Dark mode"
    if st.button(label, key="_theme_toggle", width="stretch"):
        st.session_state["theme"] = "light" if is_dark() else "dark"
        st.rerun()


# ── CSS injection ────────────────────────────────────────

def inject_theme_css() -> None:  # noqa: C901 – big but simple
    """Inject a single <style> block that themes the entire app."""
    t = tokens()
    css = f"""
<style>
/* ── Base ─────────────────────────────────────── */
.stApp {{
    background: {t["bg"]};
    color: {t["text"]};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {t["surface"]};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid {t["border"]};
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {{
    color: {t["text"]} !important;
}}

/* ── Glassmorphism card ───────────────────────── */
.glass-card {{
    background: {t["card"]};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
.glass-card:hover {{
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    transform: translateY(-1px);
}}

/* ── Metric cards ─────────────────────────────── */
.metric-card {{
    background: {t["metric_bg"]};
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid {t["metric_border"]};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: box-shadow 0.2s ease;
}}
.metric-card:hover {{
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}}
.metric-card .metric-label {{
    font-size: 0.78rem;
    color: {t["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.25rem;
}}
.metric-card .metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {t["accent"]};
    line-height: 1.2;
}}
.metric-card .metric-value.secondary {{
    color: {t["accent2"]};
}}

/* ── Badges ───────────────────────────────────── */
.lib-badge {{
    display: inline-block;
    background: {t["lib_badge_bg"]};
    color: {t["lib_badge_fg"]};
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.3rem;
}}
.ref-badge {{
    display: inline-block;
    background: {t["ref_badge_bg"]};
    color: {t["ref_badge_fg"]};
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.82rem;
    font-weight: 600;
}}

/* ── Section headers ──────────────────────────── */
.section-header {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {t["accent"]};
    border-bottom: 2px solid {t["accent"]};
    padding-bottom: 0.35rem;
    margin: 1.5rem 0 1rem;
}}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.3rem;
    background: {t["tab_bg"]};
    border-radius: 10px;
    padding: 0.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: {t["text_muted"]};
}}
.stTabs [aria-selected="true"] {{
    background: {t["accent"]} !important;
    color: #ffffff !important;
    border-radius: 8px;
}}

/* ── Data frames ──────────────────────────────── */
.stDataFrame {{
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Expanders ────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {t["surface"]};
    border-radius: 8px;
    color: {t["text"]} !important;
}}

/* ── Buttons ──────────────────────────────────── */
.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}}

/* ── Tighten top padding ──────────────────────── */
.block-container {{
    padding-top: 1.5rem;
}}

/* ── Info / warning boxes ─────────────────────── */
.stAlert {{
    border-radius: 8px;
}}

/* ── Dividers ─────────────────────────────────── */
hr {{
    border-color: {t["border"]} !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ── HTML helpers ─────────────────────────────────────────

def metric_card(label: str, value: str, *, secondary: bool = False) -> str:
    """Return HTML for a single metric card."""
    cls = "metric-value secondary" if secondary else "metric-value"
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="{cls}">{value}</div>'
        f'</div>'
    )


def lib_badge(name: str) -> str:
    return f'<span class="lib-badge">{name}</span>'


def ref_badge(name: str) -> str:
    return f'<span class="ref-badge">{name}</span>'


def glass_card(html: str) -> str:
    return f'<div class="glass-card">{html}</div>'
