/**
 * ExperimentDetail — Drill-down view for a single experiment.
 *
 * Sections (tabs):
 *  1. Overview — Key metrics per library with ranking badges
 *  2. Accuracy — Error distribution (percentile chart), all metrics table
 *  3. Performance — Latency / throughput comparison bars
 *  4. Assumptions — Alignment checklist
 */

import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";
import { fetchExperiment } from "../api/client";
import type { ExperimentResult } from "../api/types";
import Header from "../components/layout/Header";
import MetricCard from "../components/cards/MetricCard";
import {
  getPrimaryMetric,
  getAllAccuracyMetrics,
  fmtValue,
  fmtNs,
  fmtOpsS,
  libColor,
} from "../lib/analytics";

const TABS = ["Overview", "Accuracy", "Performance", "Assumptions"] as const;
type Tab = (typeof TABS)[number];

// ─── Component ───────────────────────────────────────────────────────

export default function ExperimentDetail() {
  const { runId, experiment } = useParams<{
    runId: string;
    experiment: string;
  }>();
  const [tab, setTab] = useState<Tab>("Overview");

  const {
    data: results,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["experiment", runId, experiment],
    queryFn: () => fetchExperiment(runId!, experiment!),
    enabled: !!runId && !!experiment,
  });

  if (isLoading) return <p className="text-gray-500">Loading experiment...</p>;
  if (error || !results)
    return <p className="text-red-400">Failed to load experiment.</p>;

  const ref = results[0]?.reference_library ?? "erfa";
  const mode = results[0]?.alignment?.mode ?? "common_denominator";

  return (
    <div>
      <Header
        title={experiment!.replace(/_/g, " ")}
        subtitle={`Reference: ${ref} \u2014 Mode: ${mode}`}
        actions={
          <Link
            to={`/runs/${runId}`}
            className="flex items-center gap-2 rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700"
          >
            <ArrowLeft className="h-4 w-4" />
            Run Overview
          </Link>
        }
      />

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 border-b border-gray-800">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === t
                ? "text-white border-b-2 border-orange-500"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Overview" && <OverviewTab results={results} />}
      {tab === "Accuracy" && <AccuracyTab results={results} />}
      {tab === "Performance" && <PerformanceTab results={results} />}
      {tab === "Assumptions" && <AssumptionsTab results={results} />}
    </div>
  );
}

// ─── Overview Tab ────────────────────────────────────────────────────

function OverviewTab({ results }: { results: ExperimentResult[] }) {
  // Determine primary metric
  const pm = getPrimaryMetric(results[0]?.accuracy as Record<string, unknown>);

  // Find best performers
  const accuracyValues = results
    .map((r) => {
      const m = getPrimaryMetric(r.accuracy as Record<string, unknown>);
      return { library: r.candidate_library, p99: m.stats?.p99 ?? null };
    })
    .filter((v) => v.p99 != null);

  const bestAccuracy =
    accuracyValues.length > 0
      ? accuracyValues.reduce((a, b) =>
          Math.abs(a.p99!) < Math.abs(b.p99!) ? a : b,
        ).library
      : null;

  const perfValues = results
    .map((r) => {
      const perf = r.performance as Record<string, unknown>;
      return {
        library: r.candidate_library,
        ns: (perf?.per_op_ns as number) ?? null,
      };
    })
    .filter((v) => v.ns != null);

  const bestPerf =
    perfValues.length > 0
      ? perfValues.reduce((a, b) => (a.ns! < b.ns! ? a : b)).library
      : null;

  return (
    <div className="space-y-6">
      {results.map((r) => {
        const acc = r.accuracy as Record<string, unknown>;
        const metric = getPrimaryMetric(acc);
        const s = metric.stats;
        const nan = (acc.nan_count as number) ?? 0;
        const inf = (acc.inf_count as number) ?? 0;
        const perf = r.performance as Record<string, unknown>;
        const nsOp = (perf?.per_op_ns as number) ?? null;
        const opsS = (perf?.throughput_ops_s as number) ?? null;

        const isAccBest = r.candidate_library === bestAccuracy;
        const isPerfBest = r.candidate_library === bestPerf;

        return (
          <div
            key={r.candidate_library}
            className="rounded-xl border border-gray-800 bg-gray-900/40 p-5"
          >
            <div className="flex items-center gap-3 mb-4">
              <span
                className="inline-block w-3 h-3 rounded-full"
                style={{ backgroundColor: libColor(r.candidate_library) }}
              />
              <h3 className="text-sm font-semibold text-white">
                {r.candidate_library}
              </h3>
              <span className="text-xs text-gray-500">
                vs {r.reference_library}
              </span>
              {isAccBest && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-emerald-900/60 text-emerald-300">
                  MOST ACCURATE
                </span>
              )}
              {isPerfBest && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-yellow-900/60 text-yellow-300">
                  FASTEST
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <MetricCard
                label={`${pm.label} p50`}
                value={s?.p50}
                unit={pm.unit}
                accent={isAccBest ? "green" : "default"}
              />
              <MetricCard
                label={`${pm.label} p99`}
                value={s?.p99}
                unit={pm.unit}
                accent={isAccBest ? "green" : "default"}
              />
              <MetricCard
                label={`${pm.label} max`}
                value={s?.max}
                unit={pm.unit}
              />
              <MetricCard
                label="Latency"
                value={nsOp != null ? fmtNs(nsOp) : null}
                accent={isPerfBest ? "yellow" : "default"}
              />
              <MetricCard
                label="Throughput"
                value={opsS != null ? `${fmtOpsS(opsS)} ops/s` : null}
              />
              <MetricCard
                label="NaN / Inf"
                value={`${nan} / ${inf}`}
                accent={nan + inf > 0 ? "red" : "green"}
              />
            </div>
          </div>
        );
      })}

      {/* Parity note */}
      {results[0]?.alignment?.note && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-1">
            Parity Note
          </h3>
          <p className="text-sm text-gray-300">{results[0].alignment.note}</p>
        </div>
      )}
    </div>
  );
}

// ─── Accuracy Tab ────────────────────────────────────────────────────

function AccuracyTab({ results }: { results: ExperimentResult[] }) {
  return (
    <div className="space-y-8">
      {/* Percentile comparison chart */}
      <PercentileComparisonChart results={results} />

      {/* Detailed accuracy metrics table */}
      <AccuracyMetricsTable results={results} />
    </div>
  );
}

/** Chart showing percentile distribution per library (p50, p90, p95, p99, max). */
function PercentileComparisonChart({
  results,
}: {
  results: ExperimentResult[];
}) {
  const pm = getPrimaryMetric(results[0]?.accuracy as Record<string, unknown>);
  if (!pm.key) return null;

  const percentiles = ["min", "p50", "p90", "p95", "p99", "max"] as const;
  const percentileLabels = ["Min", "p50", "p90", "p95", "p99", "Max"];

  const traces: Data[] = results.map((r) => {
    const metric = getPrimaryMetric(r.accuracy as Record<string, unknown>);
    const s = metric.stats;
    if (!s) return null;

    const values = percentiles.map((p) => (s as unknown as Record<string, number | null>)[p] ?? null);

    return {
      x: percentileLabels,
      y: values,
      name: r.candidate_library,
      type: "scatter" as const,
      mode: "lines+markers" as const,
      line: {
        color: libColor(r.candidate_library),
        width: 2.5,
      },
      marker: {
        color: libColor(r.candidate_library),
        size: 8,
      },
      hovertemplate:
        `<b>${r.candidate_library}</b><br>` +
        "%{x}: %{y:.4g} " +
        pm.unit +
        "<extra></extra>",
    } as Data;
  }).filter((t): t is Data => t !== null);

  const layout: Partial<Layout> = {
    title: {
      text: `${pm.label} Distribution by Percentile`,
      font: { color: "#e5e7eb", size: 15 },
      x: 0.01,
      xanchor: "left",
    },
    xaxis: { color: "#9ca3af", tickfont: { size: 12 } },
    yaxis: {
      title: { text: `${pm.label} (${pm.unit})`, font: { size: 12 } },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#1f2937",
      zerolinecolor: "#1f2937",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db", family: "Inter, system-ui, sans-serif" },
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d1d5db" },
      orientation: "h",
      y: -0.2,
      x: 0.5,
      xanchor: "center",
    },
    margin: { t: 50, r: 20, b: 70, l: 70 },
    height: 380,
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-2">
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
      />
      <p className="text-xs text-gray-500 px-3 pb-2">
        Lines show how error grows across percentiles. Flatter = more consistent. Lower = more accurate.
      </p>
    </div>
  );
}

/** Table showing all accuracy metrics for all libraries. */
function AccuracyMetricsTable({
  results,
}: {
  results: ExperimentResult[];
}) {
  // Collect all unique metrics across all results
  const allMetrics = results.flatMap((r) => getAllAccuracyMetrics(r));
  const metricKeys = [...new Set(allMetrics.map((m) => m.key))];

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase text-gray-400">
            <th className="px-3 py-2">Metric</th>
            <th className="px-3 py-2">Library</th>
            <th className="px-3 py-2 text-right">Mean</th>
            <th className="px-3 py-2 text-right">RMS</th>
            <th className="px-3 py-2 text-right">p50</th>
            <th className="px-3 py-2 text-right">p90</th>
            <th className="px-3 py-2 text-right">p95</th>
            <th className="px-3 py-2 text-right">p99</th>
            <th className="px-3 py-2 text-right">Max</th>
            <th className="px-3 py-2">Unit</th>
          </tr>
        </thead>
        <tbody>
          {metricKeys.flatMap((metricKey) => {
            const entries = results
              .map((r) => {
                const metrics = getAllAccuracyMetrics(r);
                const m = metrics.find((x) => x.key === metricKey);
                return m
                  ? { library: r.candidate_library, metric: m }
                  : null;
              })
              .filter(
                (e): e is { library: string; metric: (typeof allMetrics)[0] } =>
                  e !== null,
              );

            // Find best p99 for this metric
            const p99Values = entries
              .filter((e) => e.metric.stats.p99 != null)
              .map((e) => ({
                library: e.library,
                val: Math.abs(e.metric.stats.p99!),
              }));
            const bestP99Lib =
              p99Values.length > 0
                ? p99Values.reduce((a, b) => (a.val < b.val ? a : b))
                    .library
                : null;

            return entries.map((e, idx) => (
              <tr
                key={`${metricKey}-${e.library}`}
                className="border-t border-gray-800/50 hover:bg-gray-800/30"
              >
                {idx === 0 && (
                  <td
                    className="px-3 py-2 text-gray-300 font-medium"
                    rowSpan={entries.length}
                  >
                    {e.metric.label}
                  </td>
                )}
                <td className="px-3 py-2">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: libColor(e.library) }}
                    />
                    {e.library}
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.mean)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.rms)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.p50)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.p90)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.p95)}
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono ${
                    e.library === bestP99Lib
                      ? "text-emerald-300 font-semibold"
                      : "text-gray-300"
                  }`}
                >
                  {fmtStat(e.metric.stats.p99)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400">
                  {fmtStat(e.metric.stats.max)}
                </td>
                <td className="px-3 py-2 text-gray-500">{e.metric.unit}</td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Performance Tab ─────────────────────────────────────────────────

function PerformanceTab({ results }: { results: ExperimentResult[] }) {
  const bars = results
    .map((r) => {
      const perf = r.performance as Record<string, unknown>;
      const ns = perf?.per_op_ns as number | undefined;
      const ops = perf?.throughput_ops_s as number | undefined;
      if (ns == null) return null;
      return {
        library: r.candidate_library,
        ns,
        ops: ops ?? null,
      };
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);

  // Reference performance
  const refPerf = results[0]?.reference_performance as Record<string, unknown>;
  const refNs = (refPerf?.per_op_ns as number) ?? null;
  const refOps = (refPerf?.throughput_ops_s as number) ?? null;
  const refLib = results[0]?.reference_library ?? "erfa";

  if (bars.length === 0 && refNs == null) {
    return (
      <p className="text-sm text-gray-500 italic">
        No performance data available for this experiment.
      </p>
    );
  }

  // All bars including reference
  const allBars = refNs != null
    ? [{ library: refLib, ns: refNs, ops: refOps }, ...bars]
    : bars;

  // Find fastest
  const fastest = bars.length > 0
    ? bars.reduce((a, b) => (a.ns < b.ns ? a : b)).library
    : null;

  // Latency bar chart
  const latencyTrace: Data = {
    x: allBars.map((b) => b.library),
    y: allBars.map((b) => b.ns),
    type: "bar" as const,
    marker: {
      color: allBars.map((b) => libColor(b.library)),
      opacity: 0.85,
    },
    text: allBars.map((b) => fmtNs(b.ns)),
    textposition: "outside" as const,
    textfont: { size: 11, color: "#9ca3af" },
    hovertemplate: "<b>%{x}</b><br>%{y:,.0f} ns/op<extra></extra>",
  };

  const latencyLayout: Partial<Layout> = {
    title: {
      text: "Execution Time (ns/op)",
      font: { color: "#e5e7eb", size: 15 },
      x: 0.01,
      xanchor: "left",
    },
    xaxis: { color: "#9ca3af" },
    yaxis: {
      title: { text: "ns / op", font: { size: 12 } },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#1f2937",
      zerolinecolor: "#1f2937",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db", family: "Inter, system-ui, sans-serif" },
    margin: { t: 50, r: 20, b: 50, l: 70 },
    height: 350,
  };

  // Throughput bar chart
  const throughputBars = allBars.filter((b) => b.ops != null);
  let throughputTrace: Data | null = null;
  if (throughputBars.length > 0) {
    throughputTrace = {
      x: throughputBars.map((b) => b.library),
      y: throughputBars.map((b) => b.ops!),
      type: "bar" as const,
      marker: {
        color: throughputBars.map((b) => libColor(b.library)),
        opacity: 0.85,
      },
      text: throughputBars.map((b) => `${fmtOpsS(b.ops)} ops/s`),
      textposition: "outside" as const,
      textfont: { size: 11, color: "#9ca3af" },
      hovertemplate: "<b>%{x}</b><br>%{y:,.0f} ops/s<extra></extra>",
    };
  }

  const throughputLayout: Partial<Layout> = {
    title: {
      text: "Throughput (ops/s)",
      font: { color: "#e5e7eb", size: 15 },
      x: 0.01,
      xanchor: "left",
    },
    xaxis: { color: "#9ca3af" },
    yaxis: {
      title: { text: "ops / s", font: { size: 12 } },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#1f2937",
      zerolinecolor: "#1f2937",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db", family: "Inter, system-ui, sans-serif" },
    margin: { t: 50, r: 20, b: 50, l: 70 },
    height: 350,
  };

  // Speedup table
  const speedups = refNs != null
    ? bars.map((b) => ({
        library: b.library,
        speedup: refNs / b.ns,
        isFastest: b.library === fastest,
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-2">
        <Plot
          data={[latencyTrace]}
          layout={latencyLayout}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
        />
      </div>

      {throughputTrace && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-2">
          <Plot
            data={[throughputTrace]}
            layout={throughputLayout}
            config={{ responsive: true, displayModeBar: false }}
            className="w-full"
          />
        </div>
      )}

      {/* Speedup relative to reference */}
      {speedups.length > 0 && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-3">
            Speedup vs Reference ({refLib})
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {speedups.map((s) => (
              <div
                key={s.library}
                className={`rounded-lg border px-4 py-3 ${
                  s.isFastest
                    ? "border-yellow-600/40 bg-yellow-950/20"
                    : "border-gray-800"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: libColor(s.library) }}
                  />
                  <span className="text-sm font-medium text-gray-200">
                    {s.library}
                  </span>
                  {s.isFastest && (
                    <span className="text-[10px] font-bold text-yellow-300 bg-yellow-900/50 rounded-full px-1.5 py-0.5">
                      FASTEST
                    </span>
                  )}
                </div>
                <p className="mt-1 text-2xl font-bold text-white">
                  {s.speedup >= 1
                    ? `${s.speedup.toFixed(1)}x`
                    : `${(1 / s.speedup).toFixed(1)}x slower`}
                </p>
                <p className="text-xs text-gray-500">
                  {s.speedup >= 1 ? "faster" : "slower"} than {refLib}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Assumptions Tab ─────────────────────────────────────────────────

function AssumptionsTab({ results }: { results: ExperimentResult[] }) {
  const alignment = results[0]?.alignment;
  if (!alignment) {
    return (
      <p className="text-sm text-gray-500 italic">
        No alignment checklist available.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {Object.entries(alignment).map(([key, value]) => {
        if (key === "models" && typeof value === "object" && value !== null) {
          return (
            <div
              key={key}
              className="rounded-xl border border-gray-800 bg-gray-900/40 p-4"
            >
              <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
                Models per Library
              </h3>
              <div className="space-y-2">
                {Object.entries(value as Record<string, string>).map(
                  ([lib, model]) => (
                    <div key={lib} className="text-sm">
                      <span className="font-medium text-gray-300">{lib}:</span>{" "}
                      <span className="text-gray-400">{model}</span>
                    </div>
                  ),
                )}
              </div>
            </div>
          );
        }

        if (typeof value === "object" && value !== null) {
          return (
            <div
              key={key}
              className="rounded-xl border border-gray-800 bg-gray-900/40 p-4"
            >
              <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
                {key.replace(/_/g, " ")}
              </h3>
              <div className="space-y-1">
                {Object.entries(value as Record<string, string>).map(
                  ([k, v]) => (
                    <div key={k} className="text-sm">
                      <span className="text-gray-400">{k}:</span>{" "}
                      <span className="text-gray-300">{String(v)}</span>
                    </div>
                  ),
                )}
              </div>
            </div>
          );
        }

        return (
          <div
            key={key}
            className="rounded-xl border border-gray-800 bg-gray-900/40 p-4"
          >
            <h3 className="text-xs font-medium uppercase text-gray-400 mb-1">
              {key.replace(/_/g, " ")}
            </h3>
            <p className="text-sm text-gray-300">{String(value)}</p>
          </div>
        );
      })}
    </div>
  );
}

// ─── Format helper ───────────────────────────────────────────────────

function fmtStat(v: number | null | undefined): string {
  return fmtValue(v);
}
