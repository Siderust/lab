"""
Report Generator
================

Produces shareable HTML or Markdown reports from loaded ``ExperimentResult``
objects, embedding inline plots (base64 PNGs) and summary tables.

Can be invoked:
- From the Streamlit GUI ("Generate report" button).
- From the CLI::

      python -m ui.report_generator --results-dir results/ --date 2026-02-12 --experiment frame_rotation_bpn
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult

# We import plotly lazily so that this module can be tested without it
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _fmt(v: float | None, prec: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def _fig_to_b64_png(fig: "go.Figure", width: int = 800, height: int = 400) -> str:
    """Render a Plotly figure to a base64-encoded PNG string."""
    img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return base64.b64encode(img_bytes).decode()


def _fig_to_html_div(fig: "go.Figure") -> str:
    """Render a Plotly figure to an embeddable HTML <div>."""
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ──────────────────────────────────────────────────────────────────
# Plot builders (shared with app.py via import)
# ──────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

LIB_COLORS = {
    "siderust": "#2563EB",  # blue
    "astropy": "#D97706",   # amber
    "erfa": "#059669",      # green
    "libnova": "#7C3AED",   # purple
    "anise": "#DC2626",     # red
}


def _color(lib: str) -> str:
    return LIB_COLORS.get(lib, "#6B7280")


def make_accuracy_bar(results: list["ExperimentResult"], title: str = "") -> "go.Figure":
    """Grouped bar chart: p50 / p90 / p99 / max for each library."""
    fig = go.Figure()
    percentiles = ["p50", "p90", "p99", "max"]

    for r in results:
        stats = r.primary_error_stats
        vals = [getattr(stats, p, None) for p in percentiles]
        fig.add_trace(
            go.Bar(
                name=r.candidate_library,
                x=percentiles,
                y=vals,
                marker_color=_color(r.candidate_library),
                text=[_fmt(v, 3) for v in vals],
                textposition="outside",
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title=title or f"Accuracy — {unit}",
        yaxis_title=unit,
        xaxis_title="Percentile",
        barmode="group",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def make_accuracy_box(results: list["ExperimentResult"]) -> "go.Figure":
    """
    Simulated box-plot from percentile summaries.

    We don't have raw sample data, so we draw a box from the available
    percentile summaries (p50 as median, p90 as Q3-ish, etc.).
    """
    fig = go.Figure()
    for r in results:
        s = r.primary_error_stats
        if s.is_empty():
            continue
        # Use min/p50/p90/p99/max as a pseudo-box
        fig.add_trace(
            go.Box(
                name=r.candidate_library,
                q1=[s.p50 or 0],       # we'll approximate
                median=[s.p50 or 0],
                q3=[s.p90 or 0],
                lowerfence=[s.min or 0],
                upperfence=[s.max or 0],
                marker_color=_color(r.candidate_library),
                boxpoints=False,
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title=f"Accuracy Distribution — {unit}",
        yaxis_title=unit,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def make_latency_bar(results: list["ExperimentResult"]) -> "go.Figure":
    """Bar chart: per-operation latency (ns) per library, with reference line."""
    fig = go.Figure()
    libs, vals = [], []
    for r in results:
        if r.performance.per_op_ns is not None:
            libs.append(r.candidate_library)
            vals.append(r.performance.per_op_ns)

    fig.add_trace(
        go.Bar(
            x=libs,
            y=vals,
            marker_color=[_color(l) for l in libs],
            text=[f"{v:,.0f} ns" for v in vals],
            textposition="outside",
        )
    )

    # Reference performance line
    for r in results:
        if r.reference_performance.per_op_ns:
            fig.add_hline(
                y=r.reference_performance.per_op_ns,
                line_dash="dash",
                line_color=_color(r.reference_library),
                annotation_text=f"Reference ({r.reference_library}): {r.reference_performance.per_op_ns:,.0f} ns",
                annotation_position="top right",
            )
            break

    fig.update_layout(
        title="Latency — per operation (ns)",
        yaxis_title="Latency (ns/op)",
        yaxis_type="log",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def make_throughput_bar(results: list["ExperimentResult"]) -> "go.Figure":
    """Bar chart: throughput (ops/s) per library."""
    fig = go.Figure()
    libs, vals = [], []
    for r in results:
        if r.performance.throughput_ops_s is not None:
            libs.append(r.candidate_library)
            vals.append(r.performance.throughput_ops_s)

    fig.add_trace(
        go.Bar(
            x=libs,
            y=vals,
            marker_color=[_color(l) for l in libs],
            text=[f"{v:,.0f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Throughput (ops/s)",
        yaxis_title="Operations per second",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def make_pareto_scatter(results: list["ExperimentResult"]) -> "go.Figure":
    """Scatter: p99 error vs p50 latency.  One dot per library."""
    fig = go.Figure()
    for r in results:
        err = r.primary_error_stats.p99
        lat = r.performance.per_op_ns
        if err is None or lat is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[lat],
                y=[err],
                mode="markers+text",
                marker=dict(size=14, color=_color(r.candidate_library)),
                text=[r.candidate_library],
                textposition="top center",
                name=r.candidate_library,
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title="Pareto — Accuracy vs Latency",
        xaxis_title="p50 latency (ns/op) — log scale",
        yaxis_title=f"p99 {unit}",
        xaxis_type="log",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def make_gmst_era_bar(results: list["ExperimentResult"]) -> "go.Figure":
    """Side-by-side bars for GMST (arcsec) and ERA (rad) errors."""
    from plotly.subplots import make_subplots  # type: ignore

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["GMST error (arcsec)", "ERA error (rad)"],
    )

    percentiles = ["p50", "p90", "p99", "max"]
    for r in results:
        gs = r.gmst_error_arcsec
        g_vals = [getattr(gs, p, None) for p in percentiles]
        fig.add_trace(
            go.Bar(
                name=f"{r.candidate_library} — GMST",
                x=percentiles, y=g_vals,
                marker_color=_color(r.candidate_library),
                text=[_fmt(v, 6) for v in g_vals],
                textposition="outside",
                showlegend=True,
            ),
            row=1, col=1,
        )

        es = r.era_error_rad
        e_vals = [getattr(es, p, None) for p in percentiles]
        fig.add_trace(
            go.Bar(
                name=f"{r.candidate_library} — ERA",
                x=percentiles, y=e_vals,
                marker_color=_color(r.candidate_library),
                opacity=0.7,
                text=[f"{v:.2e}" if v else "—" for v in e_vals],
                textposition="outside",
                showlegend=True,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title="GMST / ERA Accuracy",
        barmode="group",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


# ──────────────────────────────────────────────────────────────────
# Summary text
# ──────────────────────────────────────────────────────────────────

def one_line_summary(results: list["ExperimentResult"]) -> str:
    """Generate a one-sentence overview."""
    parts = []
    for r in results:
        s = r.primary_error_stats
        lib = r.candidate_library
        ref = r.reference_library
        if s.p99 is not None:
            parts.append(
                f"**{lib}** matches {ref} within p99 = {_fmt(s.p99, 2)} {r.primary_error_label.split('(')[-1].rstrip(')')}"
            )
        if r.performance.per_op_ns is not None:
            parts.append(
                f"median latency = {r.performance.per_op_ns:,.0f} ns"
            )
        if r.speedup_vs_ref is not None:
            parts.append(f"speedup = {r.speedup_vs_ref:.1f}×")
    return "; ".join(parts) if parts else "No data available."


# ──────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ --bg: #f8fafc; --fg: #0f172a; --accent: #2563eb; --border: #e2e8f0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Inter, system-ui, sans-serif; background: var(--bg); color: var(--fg);
         max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.6; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid var(--accent); padding-bottom: .5rem; margin-bottom: 1.5rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 .75rem; color: var(--accent); }}
  table {{ border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .9rem; }}
  th, td {{ border: 1px solid var(--border); padding: .4rem .6rem; text-align: right; }}
  th {{ background: #f1f5f9; text-align: left; }}
  .plot {{ text-align: center; margin: 1rem 0; }}
  .plot img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }}
  pre {{ background: #f1f5f9; padding: .75rem 1rem; border-radius: 6px; overflow-x: auto; font-size: .85rem; }}
  .meta {{ font-size: .85rem; color: #64748b; }}
  .tag {{ display: inline-block; background: #e0f2fe; color: #0369a1; border-radius: 4px;
          padding: .1rem .4rem; font-size: .8rem; margin-right: .25rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {generated_at} &middot; reference: <span class="tag">{reference}</span></p>

<h2>Summary</h2>
<p>{summary_text}</p>

<h2>Top-line Metrics</h2>
{metrics_table}

<h2>Accuracy</h2>
{accuracy_section}

<h2>Performance</h2>
{performance_section}

<h2>Pareto — Accuracy vs Latency</h2>
{pareto_section}

<h2>Worst-N Outliers</h2>
{outliers_table}

<h2>Alignment Checklist (Assumptions)</h2>
<pre>{alignment_json}</pre>

<h2>Run Metadata &amp; Reproduction</h2>
<pre>{metadata_json}</pre>
<p class="meta">To reproduce this run:</p>
<pre>{repro_commands}</pre>

</body>
</html>
""")


def _metrics_html_table(results: list["ExperimentResult"]) -> str:
    """Build an HTML table summarising the key metrics for every library."""
    rows = []
    for r in results:
        s = r.primary_error_stats
        row = {
            "Library": r.candidate_library,
            "Experiment": r.experiment,
            "p50": _fmt(s.p50, 3),
            "p90": _fmt(s.p90, 3),
            "p99": _fmt(s.p99, 3),
            "max": _fmt(s.max, 3),
            "NaN": str(r.nan_count),
            "Inf": str(r.inf_count),
            "Latency (ns)": _fmt(r.performance.per_op_ns, 1),
            "Throughput": f"{r.performance.throughput_ops_s:,.0f}" if r.performance.throughput_ops_s else "—",
            "Speedup": f"{r.speedup_vs_ref:.1f}×" if r.speedup_vs_ref else "—",
        }
        rows.append(row)

    if not rows:
        return "<p>No data.</p>"

    headers = list(rows[0].keys())
    ths = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f"<td>{row[h]}</td>" for h in headers)
        body += f"<tr>{tds}</tr>\n"
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>"


def _outliers_html_table(results: list["ExperimentResult"]) -> str:
    rows = []
    for r in results:
        for wc in r.worst_cases:
            rows.append({
                "Library": r.candidate_library,
                "Case ID": wc.case_id,
                "JD(TT)": _fmt(wc.jd_tt, 6) if wc.jd_tt else "—",
                "Error (mas)": _fmt(wc.angular_error_mas, 3),
            })
    if not rows:
        return "<p>No outlier data recorded.</p>"

    headers = list(rows[0].keys())
    ths = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f"<td>{row[h]}</td>" for h in headers)
        body += f"<tr>{tds}</tr>\n"
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>"


def generate_html_report(
    results: list["ExperimentResult"],
    output_path: Path | None = None,
    embed_plots: bool = True,
) -> str:
    """
    Build an HTML report string.

    If *output_path* is given, also write it to disk (creating dirs as needed).
    If *embed_plots* is True and Plotly + kaleido are available, embed PNG plots
    as base64 ``<img>`` tags.  Otherwise, fall back to interactive divs.
    """
    if not results:
        return "<html><body><p>No results to report.</p></body></html>"

    experiment = results[0].experiment
    reference = results[0].reference_library
    title = f"Lab Report — {experiment}"

    # ── plots ──
    accuracy_html = ""
    performance_html = ""
    pareto_html = ""

    if HAS_PLOTLY:
        def _embed(fig: "go.Figure") -> str:
            try:
                b64 = _fig_to_b64_png(fig)
                return f'<div class="plot"><img src="data:image/png;base64,{b64}"></div>'
            except Exception:
                return _fig_to_html_div(fig)

        if experiment == "frame_rotation_bpn":
            accuracy_html = _embed(make_accuracy_bar(results))
        elif experiment == "gmst_era":
            accuracy_html = _embed(make_gmst_era_bar(results))

        perf_results = [r for r in results if r.has_performance]
        if perf_results:
            performance_html = _embed(make_latency_bar(perf_results))
            performance_html += _embed(make_throughput_bar(perf_results))

            pareto_results = [r for r in results if r.has_performance and r.has_accuracy]
            if pareto_results:
                pareto_html = _embed(make_pareto_scatter(pareto_results))

    html = _HTML_TEMPLATE.format(
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        reference=reference,
        summary_text=one_line_summary(results),
        metrics_table=_metrics_html_table(results),
        accuracy_section=accuracy_html or "<p>No accuracy plots available (install plotly + kaleido).</p>",
        performance_section=performance_html or "<p>No performance data.</p>",
        pareto_section=pareto_html or "<p>Not enough data for Pareto plot.</p>",
        outliers_table=_outliers_html_table(results),
        alignment_json=json.dumps(results[0].alignment, indent=2),
        metadata_json=json.dumps(results[0].run_metadata, indent=2),
        repro_commands=_repro_block(results[0]),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def _repro_block(r: "ExperimentResult") -> str:
    meta = r.run_metadata
    shas = meta.get("git_shas", {})
    lines = [
        f"# Run date: {meta.get('date', '?')}",
        f"# OS: {meta.get('os', '?')}  CPU: {meta.get('cpu', '?')}",
    ]
    for name, sha in shas.items():
        lines.append(f"# {name} SHA: {sha}")
    lines += [
        "",
        "cd /path/to/siderust/lab",
        "git submodule update --init --recursive",
        f"bash run.sh  # or: python3 pipeline/orchestrator.py --experiment {r.experiment} --n {r.input_count} --seed {r.seed or 42}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Markdown report (lighter-weight alternative)
# ──────────────────────────────────────────────────────────────────

def generate_markdown_report(results: list["ExperimentResult"]) -> str:
    """Generate a Markdown report (no embedded plots)."""
    if not results:
        return "# No results\n"

    experiment = results[0].experiment
    lines = [
        f"# Lab Report — {experiment}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Reference: **{results[0].reference_library}**",
        "",
        "## Summary",
        "",
        one_line_summary(results),
        "",
        "## Top-line Metrics",
        "",
    ]

    # Metrics table
    unit = results[0].primary_error_label
    hdr = f"| Library | {unit} p50 | p90 | p99 | max | NaN | Inf | Latency (ns) | Throughput | Speedup |"
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines += [hdr, sep]

    for r in results:
        s = r.primary_error_stats
        throughput = f"{r.performance.throughput_ops_s:,.0f}" if r.performance.throughput_ops_s else "—"
        speedup = f"{r.speedup_vs_ref:.1f}×" if r.speedup_vs_ref else "—"
        lines.append(
            f"| {r.candidate_library} "
            f"| {_fmt(s.p50, 3)} | {_fmt(s.p90, 3)} | {_fmt(s.p99, 3)} | {_fmt(s.max, 3)} "
            f"| {r.nan_count} | {r.inf_count} "
            f"| {_fmt(r.performance.per_op_ns, 1)} "
            f"| {throughput} | {speedup} |"
        )

    # Worst cases
    lines += ["", "## Worst-N Outliers", ""]
    lines.append("| Library | Case | JD(TT) | Error (mas) |")
    lines.append("|---|---|---|---|")
    for r in results:
        for wc in r.worst_cases[:5]:
            lines.append(
                f"| {r.candidate_library} | {wc.case_id} "
                f"| {_fmt(wc.jd_tt, 6)} | {_fmt(wc.angular_error_mas, 3)} |"
            )

    # Alignment
    lines += [
        "", "## Alignment Checklist", "",
        "```json",
        json.dumps(results[0].alignment, indent=2),
        "```",
    ]

    # Metadata
    lines += [
        "", "## Run Metadata", "",
        "```json",
        json.dumps(results[0].run_metadata, indent=2),
        "```",
    ]

    # Repro
    lines += [
        "", "## Reproduction", "",
        "```bash",
        _repro_block(results[0]),
        "```",
    ]

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lab report")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--date", required=True, help="Run date folder name, e.g. 2026-02-12")
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. frame_rotation_bpn")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--format", choices=["html", "md", "both"], default="both")
    args = parser.parse_args()

    # Import loader
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ui.results_loader import RunInfo, load_results

    run = RunInfo(
        date=args.date,
        experiment=args.experiment,
        path=args.results_dir / args.date / args.experiment,
    )
    results = load_results(run)
    if not results:
        print(f"No results found in {run.path}")
        return

    out_dir = args.output_dir / args.date / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("html", "both"):
        html_path = out_dir / "index.html"
        generate_html_report(results, output_path=html_path)
        print(f"✓ Wrote {html_path}")

    if args.format in ("md", "both"):
        md_path = out_dir / "report.md"
        md_path.write_text(generate_markdown_report(results))
        print(f"✓ Wrote {md_path}")


if __name__ == "__main__":
    main()
