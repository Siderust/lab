"""Overview tab — key metrics, alignment checklist, run metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from ui.report_generator import color, fmt, one_line_summary
from ui.theme import glass_card, lib_badge, metric_card, ref_badge, is_dark

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


# ──────────────────────────────────────────────────────────
# Experiment descriptions
# ──────────────────────────────────────────────────────────
EXPERIMENT_DESCRIPTIONS: dict[str, str] = {
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


def render(results: list["ExperimentResult"], experiment: str) -> None:
    desc = EXPERIMENT_DESCRIPTIONS.get(experiment, "")
    if desc:
        st.markdown(desc)

    # One-line summary
    st.info(one_line_summary(results))

    # ── Key metrics in responsive grid ──────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    # Use max 3 columns to avoid unreadable narrow cards
    n_cols = min(len(results), 3)
    for chunk_start in range(0, len(results), n_cols):
        chunk = results[chunk_start : chunk_start + n_cols]
        cols = st.columns(n_cols)
        for col, r in zip(cols, chunk):
            with col:
                st.markdown(
                    f'{lib_badge(r.candidate_library)} vs {ref_badge(r.reference_library)}',
                    unsafe_allow_html=True,
                )

                s = r.primary_error_stats
                label = r.primary_error_label

                cards_html = ""
                if s.p50 is not None:
                    cards_html += metric_card(f"{label} — p50", fmt(s.p50, 3))
                if s.p99 is not None:
                    cards_html += metric_card(f"{label} — p99", fmt(s.p99, 3))
                if s.max is not None:
                    cards_html += metric_card(f"{label} — max", fmt(s.max, 3))
                if r.performance.per_op_ns is not None:
                    cards_html += metric_card(
                        "Latency (ns/op)",
                        f"{r.performance.per_op_ns:,.0f}",
                        secondary=True,
                    )
                if r.speedup_vs_ref is not None:
                    cards_html += metric_card(
                        "Speedup vs reference",
                        f"{r.speedup_vs_ref:.1f}×",
                        secondary=True,
                    )
                cards_html += metric_card("NaN / Inf", f"{r.nan_count} / {r.inf_count}")

                st.markdown(glass_card(cards_html), unsafe_allow_html=True)

    # ── Alignment checklist ─────────────────────────────
    st.markdown('<div class="section-header">Alignment Checklist</div>', unsafe_allow_html=True)
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

    # ── Run metadata ────────────────────────────────────
    st.markdown('<div class="section-header">Run Metadata</div>', unsafe_allow_html=True)
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

    st.markdown(
        f"**Input count:** {results[0].input_count}  ·  **Seed:** {results[0].seed}"
    )
