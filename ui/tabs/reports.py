"""Reports tab — generate + preview + save HTML / Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from ui.report_generator import generate_html_report, generate_markdown_report

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def render(
    results: list["ExperimentResult"],
    experiment: str,
    selected_date: str,
    repo_root: Path,
) -> None:
    st.markdown(
        "Create a shareable HTML or Markdown report for the current selection.  "
        "Reports include all metrics, plots, the alignment checklist, and reproduction commands."
    )

    report_format = st.radio(
        "Report format",
        ["HTML (with embedded plots)", "Markdown (text only)"],
        horizontal=True,
        key="report_format",
    )

    col_gen, col_save = st.columns([1, 1])

    with col_gen:
        if st.button("🚀 Generate report", type="primary", width="stretch"):
            with st.spinner("Generating report…"):
                if "HTML" in report_format:
                    try:
                        content = generate_html_report(results)
                        st.session_state["report_content"] = content
                        st.session_state["report_ext"] = "html"
                        st.session_state["report_mime"] = "text/html"
                    except Exception as e:
                        st.error(f"Error generating HTML report: {e}")
                        st.info("Falling back to Markdown (install `kaleido` for PNG embedding).")
                        content = generate_markdown_report(results)
                        st.session_state["report_content"] = content
                        st.session_state["report_ext"] = "md"
                        st.session_state["report_mime"] = "text/markdown"
                else:
                    content = generate_markdown_report(results)
                    st.session_state["report_content"] = content
                    st.session_state["report_ext"] = "md"
                    st.session_state["report_mime"] = "text/markdown"

    with col_save:
        # Only enable save if a report has been generated
        has_report = "report_content" in st.session_state
        if st.button(
            "💾 Save to reports/ directory",
            width="stretch",
            disabled=not has_report,
        ):
            reports_dir = repo_root / "reports" / selected_date / experiment
            reports_dir.mkdir(parents=True, exist_ok=True)

            content = st.session_state["report_content"]
            ext = st.session_state["report_ext"]

            if ext == "html":
                out_path = reports_dir / "index.html"
            else:
                out_path = reports_dir / "report.md"

            out_path.write_text(content)
            st.success(f"✓ Saved to `{out_path.relative_to(repo_root)}`")

    # Preview + download
    if "report_content" in st.session_state:
        ext = st.session_state["report_ext"]
        content = st.session_state["report_content"]
        mime = st.session_state["report_mime"]

        st.divider()
        st.download_button(
            f"⬇ Download report (.{ext})",
            data=content,
            file_name=f"{experiment}_report.{ext}",
            mime=mime,
            key="dl_report",
        )

        st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
        if ext == "html":
            st.components.v1.html(content, height=800, scrolling=True)
        else:
            st.markdown(content)
