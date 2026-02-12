"""
Report Generator
================

Produces shareable HTML or Markdown reports from loaded ``ExperimentResult``
objects, embedding inline plots (base64 PNGs) and summary tables.

Can be invoked:
- From the Streamlit GUI ("Generate report" button).
- From the CLI::

      python -m ui.report_generator --results-dir results/ --date 2026-02-12 --experiment frame_rotation_bpn

Public API (also imported by app / tab modules)
------------------------------------------------
- ``fmt(v, prec)``          — safe float format
- ``color(lib)``            — per-library colour hex
- ``LIB_COLORS``            — colour map
- ``make_accuracy_bar()``   — grouped bar chart
- ``make_accuracy_box()``   — pseudo-box plot
- ``make_latency_bar()``    — latency bar chart
- ``make_throughput_bar()`` — throughput bar chart
- ``make_pareto_scatter()`` — accuracy-vs-latency scatter
- ``make_gmst_era_bar()``   — side-by-side GMST/ERA bars
- ``make_trend_chart()``    — cross-run trend line chart
- ``one_line_summary()``    — 1-sentence overview text
- ``generate_html_report()``
- ``generate_markdown_report()``
"""

from __future__ import annotations

import argparse
import base64
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult

# Lazy Plotly import
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Repo root (for repro block) ─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def fmt(v: float | None, prec: int = 2) -> str:
    """Safe number formatter — returns "—" for None."""
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


# Back-compat alias (old code imported the underscore name)
_fmt = fmt


def _fig_to_b64_png(fig: "go.Figure", width: int = 800, height: int = 400) -> str:
    """Render a Plotly figure to a base64-encoded PNG string."""
    img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return base64.b64encode(img_bytes).decode()


def _fig_to_html_div(fig: "go.Figure") -> str:
    """Render a Plotly figure to an embeddable HTML <div>."""
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ──────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────

LIB_COLORS = {
    "siderust": "#3b82f6",  # blue
    "astropy":  "#f59e0b",  # amber
    "erfa":     "#22c55e",  # green
    "libnova":  "#a855f7",  # purple
    "anise":    "#ef4444",  # red
}


def color(lib: str) -> str:
    """Return the hex colour for *lib*, grey fallback."""
    return LIB_COLORS.get(lib, "#6b7280")


# Back-compat alias
_color = color


# ──────────────────────────────────────────────────────────
# Layout defaults (theme-aware)
# ──────────────────────────────────────────────────────────

def _layout(dark: bool = True, *, extra: dict | None = None) -> dict:
    """Plotly layout defaults for the requested theme."""
    base = dict(
        template="plotly_dark" if dark else "plotly_white",
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=13,
            color="#e2e8f0" if dark else "#0f172a",
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)" if dark else "#ffffff",
    )
    if extra:
        base.update(extra)
    return base


# Legacy constant (some old imports use this directly)
PLOTLY_LAYOUT_DEFAULTS = _layout(dark=True)


# ──────────────────────────────────────────────────────────
# Plot builders
# ──────────────────────────────────────────────────────────

def make_accuracy_bar(
    results: list["ExperimentResult"],
    title: str = "",
    *,
    dark: bool = True,
) -> "go.Figure":
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
                marker_color=color(r.candidate_library),
                text=[fmt(v, 3) for v in vals],
                textposition="outside",
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title=title or f"Accuracy — {unit}",
        yaxis_title=unit,
        xaxis_title="Percentile",
        barmode="group",
        **_layout(dark),
    )
    return fig


def make_accuracy_box(
    results: list["ExperimentResult"],
    *,
    dark: bool = True,
) -> "go.Figure":
    """
    Pseudo-box plot from percentile summaries.

    Uses min→lower fence, p50→Q1, p90→median (visual centre),
    p95→Q3, max→upper fence.  This gives a visible IQR instead
    of a degenerate zero-height box.
    """
    fig = go.Figure()
    for r in results:
        s = r.primary_error_stats
        if s.is_empty():
            continue
        fig.add_trace(
            go.Box(
                name=r.candidate_library,
                lowerfence=[s.min or 0],
                q1=[s.p50 or 0],
                median=[s.p90 or s.p50 or 0],
                q3=[s.p95 or s.p99 or 0],
                upperfence=[s.max or 0],
                marker_color=color(r.candidate_library),
                fillcolor=color(r.candidate_library),
                opacity=0.55,
                boxpoints=False,
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title=f"Accuracy Distribution — {unit}",
        yaxis_title=unit,
        **_layout(dark),
    )
    return fig


def make_latency_bar(
    results: list["ExperimentResult"],
    *,
    dark: bool = True,
) -> "go.Figure":
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
            marker_color=[color(lib) for lib in libs],
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
                line_color=color(r.reference_library),
                annotation_text=(
                    f"Reference ({r.reference_library}): "
                    f"{r.reference_performance.per_op_ns:,.0f} ns"
                ),
                annotation_position="top right",
            )
            break

    fig.update_layout(
        title="Latency — per operation (ns)",
        yaxis_title="Latency (ns/op)",
        yaxis_type="log",
        **_layout(dark),
    )
    return fig


def make_throughput_bar(
    results: list["ExperimentResult"],
    *,
    dark: bool = True,
) -> "go.Figure":
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
            marker_color=[color(lib) for lib in libs],
            text=[f"{v:,.0f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Throughput (ops/s)",
        yaxis_title="Operations per second",
        **_layout(dark),
    )
    return fig


def make_pareto_scatter(
    results: list["ExperimentResult"],
    *,
    dark: bool = True,
) -> "go.Figure":
    """Scatter: p99 error vs latency.  One dot per library."""
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
                marker=dict(
                    size=16,
                    color=color(r.candidate_library),
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                text=[r.candidate_library],
                textposition="top center",
                name=r.candidate_library,
            )
        )

    unit = results[0].primary_error_label if results else "Error"
    fig.update_layout(
        title="Pareto — Accuracy vs Latency",
        xaxis_title="Latency (ns/op) — log scale",
        yaxis_title=f"p99 {unit}",
        xaxis_type="log",
        **_layout(dark),
    )
    return fig


def make_gmst_era_bar(
    results: list["ExperimentResult"],
    *,
    dark: bool = True,
) -> "go.Figure":
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
                marker_color=color(r.candidate_library),
                text=[fmt(v, 6) for v in g_vals],
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
                marker_color=color(r.candidate_library),
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
        **_layout(dark),
    )
    return fig


def make_trend_chart(
    trend_data: list[dict],
    *,
    metric: str = "p99",
    dark: bool = True,
) -> "go.Figure":
    """
    Line chart: accuracy metric over multiple run dates.

    *trend_data* is a list of dicts:
        ``{"date": "2026-02-12", "library": "siderust", "p99": 83.5, "latency_ns": 1219}``
    """
    from plotly.subplots import make_subplots  # type: ignore

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Accuracy ({metric}) over runs", "Latency (ns/op) over runs"],
        vertical_spacing=0.12,
    )

    # Group by library
    libs: dict[str, list[dict]] = {}
    for d in trend_data:
        libs.setdefault(d["library"], []).append(d)

    for lib, points in libs.items():
        points.sort(key=lambda p: p["date"])
        dates = [p["date"] for p in points]
        vals = [p.get(metric) for p in points]
        lats = [p.get("latency_ns") for p in points]

        fig.add_trace(
            go.Scatter(
                x=dates, y=vals,
                mode="lines+markers",
                name=lib,
                line=dict(color=color(lib), width=2),
                marker=dict(size=8),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=lats,
                mode="lines+markers",
                name=lib,
                line=dict(color=color(lib), width=2, dash="dot"),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ns)", row=2, col=1)
    fig.update_xaxes(title_text="Run date", row=2, col=1)
    fig.update_layout(
        height=550,
        **_layout(dark),
    )
    return fig


# ──────────────────────────────────────────────────────────
# Summary text
# ──────────────────────────────────────────────────────────

def one_line_summary(results: list["ExperimentResult"]) -> str:
    """Generate a one-sentence overview."""
    parts = []
    for r in results:
        s = r.primary_error_stats
        lib = r.candidate_library
        ref = r.reference_library
        if s.p99 is not None:
            unit = r.primary_error_label.split("(")[-1].rstrip(")")
            parts.append(f"**{lib}** matches {ref} within p99 = {fmt(s.p99, 2)} {unit}")
        if r.performance.per_op_ns is not None:
            parts.append(f"median latency = {r.performance.per_op_ns:,.0f} ns")
        if r.speedup_vs_ref is not None:
            parts.append(f"speedup = {r.speedup_vs_ref:.1f}×")
    return "; ".join(parts) if parts else "No data available."


# ──────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────

_HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ --bg: #0f172a; --fg: #e2e8f0; --accent: #3b82f6; --border: rgba(255,255,255,0.08);
           --surface: rgba(30,41,59,0.8); }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Inter, system-ui, sans-serif; background: var(--bg); color: var(--fg);
         max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.6; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid var(--accent); padding-bottom: .5rem;
       margin-bottom: 1.5rem; color: var(--accent); }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 .75rem; color: var(--accent); }}
  table {{ border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .9rem; }}
  th, td {{ border: 1px solid var(--border); padding: .4rem .6rem; text-align: right; }}
  th {{ background: var(--surface); text-align: left; }}
  .plot {{ text-align: center; margin: 1rem 0; }}
  .plot img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 8px; }}
  pre {{ background: var(--surface); padding: .75rem 1rem; border-radius: 8px;
        overflow-x: auto; font-size: .85rem; }}
  .meta {{ font-size: .85rem; color: #94a3b8; }}
  .tag {{ display: inline-block; background: rgba(59,130,246,.18); color: #93c5fd;
          border-radius: 6px; padding: .1rem .5rem; font-size: .8rem; margin-right: .25rem; }}
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
    rows = []
    for r in results:
        s = r.primary_error_stats
        row = {
            "Library": r.candidate_library,
            "Experiment": r.experiment,
            "p50": fmt(s.p50, 3),
            "p90": fmt(s.p90, 3),
            "p99": fmt(s.p99, 3),
            "max": fmt(s.max, 3),
            "NaN": str(r.nan_count),
            "Inf": str(r.inf_count),
            "Latency (ns)": fmt(r.performance.per_op_ns, 1),
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
        label = r.primary_error_label
        for wc in r.worst_cases:
            rows.append({
                "Library": r.candidate_library,
                "Case ID": wc.case_id,
                "JD(TT)": fmt(wc.jd_tt, 6) if wc.jd_tt else "—",
                f"{label}": fmt(wc.angular_error_mas, 3),
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
    """Build an HTML report string.  Optionally write to *output_path*."""
    if not results:
        return "<html><body><p>No results to report.</p></body></html>"

    experiment = results[0].experiment
    reference = results[0].reference_library
    title = f"Lab Report — {experiment}"

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
            accuracy_html = _embed(make_accuracy_bar(results, dark=False))
        elif experiment == "gmst_era":
            accuracy_html = _embed(make_gmst_era_bar(results, dark=False))

        perf_results = [r for r in results if r.has_performance]
        if perf_results:
            performance_html = _embed(make_latency_bar(perf_results, dark=False))
            performance_html += _embed(make_throughput_bar(perf_results, dark=False))
            pareto_results = [r for r in results if r.has_performance and r.has_accuracy]
            if pareto_results:
                pareto_html = _embed(make_pareto_scatter(pareto_results, dark=False))

    html = _HTML_TEMPLATE.format(
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        reference=reference,
        summary_text=one_line_summary(results),
        metrics_table=_metrics_html_table(results),
        accuracy_section=accuracy_html or "<p>No accuracy plots (install plotly + kaleido).</p>",
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
        f"cd {_REPO_ROOT}",
        "git submodule update --init --recursive",
        (
            f"bash run.sh  # or: python3 pipeline/orchestrator.py"
            f" --experiment {r.experiment} --n {r.input_count} --seed {r.seed or 42}"
        ),
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────

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
            f"| {fmt(s.p50, 3)} | {fmt(s.p90, 3)} | {fmt(s.p99, 3)} | {fmt(s.max, 3)} "
            f"| {r.nan_count} | {r.inf_count} "
            f"| {fmt(r.performance.per_op_ns, 1)} "
            f"| {throughput} | {speedup} |"
        )

    lines += ["", "## Worst-N Outliers", ""]
    lines.append("| Library | Case | JD(TT) | Error |")
    lines.append("|---|---|---|---|")
    for r in results:
        for wc in r.worst_cases[:5]:
            lines.append(
                f"| {r.candidate_library} | {wc.case_id} "
                f"| {fmt(wc.jd_tt, 6)} | {fmt(wc.angular_error_mas, 3)} |"
            )

    lines += [
        "", "## Alignment Checklist", "",
        "```json", json.dumps(results[0].alignment, indent=2), "```",
    ]
    lines += [
        "", "## Run Metadata", "",
        "```json", json.dumps(results[0].run_metadata, indent=2), "```",
    ]
    lines += [
        "", "## Reproduction", "",
        "```bash", _repro_block(results[0]), "```",
    ]

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lab report")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--format", choices=["html", "md", "both"], default="both")
    args = parser.parse_args()

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
