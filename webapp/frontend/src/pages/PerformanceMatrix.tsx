/**
 * PerformanceMatrix — 2D FROM×TO heatmap grids comparing siderust's
 * coordinate-transform performance and accuracy against each library.
 *
 * Two tabs: Performance (speedup heatmap) | Accuracy (error heatmap).
 * Library selector: vs ERFA, vs Astropy, vs libnova, vs ANISE.
 * Grid: rows = source frames, columns = target frames.
 */

import { useState, useMemo } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Zap, Target } from "lucide-react";
import { fetchPerformanceMatrix } from "../api/client";
import type { PerfMatrixCell, AccuracyMatrixCell } from "../api/types";
import Header from "../components/layout/Header";

// ─── Short display names for compact grid headers ────────────────────

const FRAME_SHORT: Record<string, string> = {
  ICRS: "ICRS",
  EqMeanJ2000: "Eq M J2000",
  EqMeanOfDate: "Eq M Date",
  EqTrueOfDate: "Eq T Date",
  EclMeanJ2000: "Ecl M J2000",
  EclTrueOfDate: "Ecl T Date",
  Horizontal: "Horiz",
  GCRS: "GCRS",
  CIRS: "CIRS",
  TIRS: "TIRS",
  ITRF: "ITRF",
  Galactic: "Galactic",
};

const EXPERIMENT_LABELS: Record<string, string> = {
  frame_bias: "Frame Bias",
  precession: "Precession",
  nutation: "Nutation",
  frame_rotation_bpn: "BPN Rotation",
  icrs_ecl_j2000: "ICRS → Ecl J2000",
  icrs_ecl_tod: "ICRS → Ecl True",
  equ_ecl: "Equ → Ecl",
  equ_horizontal: "Equ → Horiz",
  horiz_to_equ: "Horiz → Equ",
  inv_frame_bias: "Inv Frame Bias",
  inv_precession: "Inv Precession",
  inv_nutation: "Inv Nutation",
  inv_bpn: "Inv BPN",
  inv_icrs_ecl_j2000: "Inv ICRS → Ecl J2000",
  obliquity: "Obliquity",
  inv_obliquity: "Inv Obliquity",
  bias_precession: "Bias+Precession",
  inv_bias_precession: "Inv Bias+Precession",
  precession_nutation: "Prec+Nutation",
  inv_precession_nutation: "Inv Prec+Nutation",
  inv_icrs_ecl_tod: "Inv ICRS → Ecl True",
  inv_equ_ecl: "Inv Equ → Ecl",
};

const COMPARISON_LIBS = [
  { key: "siderust_vs_erfa", label: "vs ERFA (C)" },
  { key: "siderust_vs_astropy", label: "vs Astropy (Python)" },
  { key: "siderust_vs_libnova", label: "vs libnova (C)" },
  { key: "siderust_vs_anise", label: "vs ANISE (Rust)" },
];

type TabMode = "perf" | "accuracy";

// ─── Color helpers ───────────────────────────────────────────────────

function speedupBg(speedup: number): string {
  if (speedup >= 10) return "bg-emerald-800/80";
  if (speedup >= 5) return "bg-emerald-800/60";
  if (speedup >= 2) return "bg-emerald-900/60";
  if (speedup >= 1.2) return "bg-emerald-900/30";
  if (speedup >= 0.8) return "bg-yellow-900/40";
  if (speedup >= 0.5) return "bg-red-900/40";
  return "bg-red-900/60";
}

function speedupText(speedup: number): string {
  if (speedup >= 5) return "text-emerald-200 font-bold";
  if (speedup >= 2) return "text-emerald-300 font-semibold";
  if (speedup >= 1.2) return "text-emerald-400";
  if (speedup >= 0.8) return "text-yellow-300";
  if (speedup >= 0.5) return "text-red-400";
  return "text-red-300 font-semibold";
}

function accuracyBg(mas: number): string {
  if (mas < 0.01) return "bg-emerald-800/80";
  if (mas < 0.1) return "bg-emerald-900/60";
  if (mas < 1) return "bg-emerald-900/30";
  if (mas < 10) return "bg-yellow-900/30";
  if (mas < 100) return "bg-orange-900/40";
  if (mas < 1000) return "bg-red-900/40";
  return "bg-red-900/60";
}

function accuracyText(mas: number): string {
  if (mas < 0.1) return "text-emerald-200 font-bold";
  if (mas < 1) return "text-emerald-300";
  if (mas < 10) return "text-yellow-300";
  if (mas < 100) return "text-orange-300";
  return "text-red-300 font-semibold";
}

function formatMas(mas: number): string {
  if (mas < 0.001) return `${(mas * 1000).toFixed(1)} µas`;
  if (mas < 1) return `${mas.toFixed(3)} mas`;
  if (mas < 1000) return `${mas.toFixed(1)} mas`;
  if (mas < 3600_000) return `${(mas / 1000).toFixed(1)} as`;
  return `${(mas / 3600_000).toFixed(1)}°`;
}

function formatNs(ns: number): string {
  if (ns < 1000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1000).toFixed(1)} µs`;
  return `${(ns / 1_000_000).toFixed(1)} ms`;
}

// ─── Cell rendering ──────────────────────────────────────────────────

function PerfCell({ cell }: { cell: PerfMatrixCell | undefined }) {
  if (!cell || cell.status === "no_experiment")
    return (
      <td className="border border-gray-800/50 bg-gray-900/30 text-center text-[10px] text-gray-700 p-1">
        —
      </td>
    );

  if (cell.status === "not_implemented")
    return (
      <td className="border border-gray-800/50 bg-gray-950/50 text-center text-[10px] text-gray-700 p-1 italic">
        n/i
      </td>
    );

  if (cell.status === "skipped")
    return (
      <td
        className="border border-gray-800/50 bg-gray-900/40 text-center text-[10px] text-gray-600 p-1"
        title={`${EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment}: skipped by this library`}
      >
        skip
      </td>
    );

  if (cell.status === "no_data")
    return (
      <td
        className="border border-gray-800/50 bg-gray-900/30 text-center text-[10px] text-gray-600 p-1"
        title={`${EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment}: no data yet`}
      >
        · · ·
      </td>
    );

  // available
  const sp = cell.speedup!;
  const tooltip = [
    EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment,
    `Speedup: ${sp.toFixed(2)}×`,
    `siderust: ${formatNs(cell.siderust_ns!)}`,
    `other: ${formatNs(cell.other_ns!)}`,
  ].join("\n");

  return (
    <td
      className={`border border-gray-800/50 text-center p-1 cursor-default transition-colors ${speedupBg(sp)}`}
      title={tooltip}
    >
      <div className={`font-mono text-xs leading-tight ${speedupText(sp)}`}>
        {sp.toFixed(1)}×
      </div>
      <div className="text-[9px] opacity-50 leading-tight mt-0.5">
        {formatNs(cell.siderust_ns!)}
      </div>
    </td>
  );
}

function AccCell({ cell }: { cell: AccuracyMatrixCell | undefined }) {
  if (!cell || cell.status === "no_experiment")
    return (
      <td className="border border-gray-800/50 bg-gray-900/30 text-center text-[10px] text-gray-700 p-1">
        —
      </td>
    );

  if (cell.status === "not_implemented")
    return (
      <td className="border border-gray-800/50 bg-gray-950/50 text-center text-[10px] text-gray-700 p-1 italic">
        n/i
      </td>
    );

  if (cell.status === "skipped")
    return (
      <td
        className="border border-gray-800/50 bg-gray-900/40 text-center text-[10px] text-gray-600 p-1"
        title={`${EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment}: skipped by this library`}
      >
        skip
      </td>
    );

  if (cell.status === "no_data")
    return (
      <td
        className="border border-gray-800/50 bg-gray-900/30 text-center text-[10px] text-gray-600 p-1"
        title={`${EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment}: no data yet`}
      >
        · · ·
      </td>
    );

  // available
  const p50 = cell.p50_mas!;
  const tooltip = [
    EXPERIMENT_LABELS[cell.experiment ?? ""] ?? cell.experiment,
    `p50: ${formatMas(p50)}`,
    `p99: ${formatMas(cell.p99_mas!)}`,
    `max: ${formatMas(cell.max_mas!)}`,
  ].join("\n");

  return (
    <td
      className={`border border-gray-800/50 text-center p-1 cursor-default transition-colors ${accuracyBg(p50)}`}
      title={tooltip}
    >
      <div className={`font-mono text-xs leading-tight ${accuracyText(p50)}`}>
        {formatMas(p50)}
      </div>
      <div className="text-[9px] opacity-50 leading-tight mt-0.5">
        p99: {formatMas(cell.p99_mas!)}
      </div>
    </td>
  );
}

// ─── Main component ──────────────────────────────────────────────────

export default function PerformanceMatrix() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<TabMode>("perf");
  const [compLib, setCompLib] = useState(COMPARISON_LIBS[0].key);
  const [showExtended, setShowExtended] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["performance-matrix", runId],
    queryFn: () => fetchPerformanceMatrix(runId!),
    enabled: !!runId,
  });

  // Derive visible frames
  const frames = useMemo(() => {
    if (!data) return [];
    return showExtended ? data.frames : data.core_frames;
  }, [data, showExtended]);

  // Get the active matrix
  const perfGrid = data?.perf_matrix?.[compLib];
  const accGrid = data?.accuracy_matrix?.[compLib];
  const activeGrid = tab === "perf" ? perfGrid : accGrid;

  // Compute summary stats for performance tab
  const perfStats = useMemo(() => {
    if (!perfGrid) return null;
    const speedups: number[] = [];
    for (const cell of Object.values(perfGrid)) {
      const c = cell as PerfMatrixCell;
      if (c.status === "available" && c.speedup != null) speedups.push(c.speedup);
    }
    if (speedups.length === 0) return null;
    const geo = Math.exp(speedups.reduce((s, v) => s + Math.log(v), 0) / speedups.length);
    const wins = speedups.filter((s) => s > 1).length;
    return {
      geo: geo.toFixed(2),
      min: Math.min(...speedups).toFixed(1),
      max: Math.max(...speedups).toFixed(1),
      wins,
      total: speedups.length,
    };
  }, [perfGrid]);

  if (isLoading) return <p className="text-gray-500 p-8">Loading performance matrix…</p>;
  if (error || !data)
    return <p className="text-red-400 p-8">Failed to load performance matrix for run {runId}.</p>;

  const hasAnyPerfData =
    perfGrid && Object.values(perfGrid).some((c) => (c as PerfMatrixCell).status === "available");
  const hasAnyAccData =
    accGrid && Object.values(accGrid).some((c) => (c as AccuracyMatrixCell).status === "available");
  const hasAnyData = hasAnyPerfData || hasAnyAccData;

  return (
    <div className="space-y-6">
      <Header
        title="Performance Matrix"
        subtitle={`Run ${data.run_id} — FROM × TO coordinate transform comparison`}
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

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Tab switcher */}
        <div className="flex rounded-lg border border-gray-700 overflow-hidden text-sm">
          <button
            onClick={() => setTab("perf")}
            className={`flex items-center gap-1.5 px-4 py-2 transition-colors ${
              tab === "perf"
                ? "bg-orange-600/20 text-orange-300 font-semibold"
                : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            <Zap className="h-3.5 w-3.5" />
            Performance
          </button>
          <button
            onClick={() => setTab("accuracy")}
            className={`flex items-center gap-1.5 px-4 py-2 transition-colors ${
              tab === "accuracy"
                ? "bg-blue-600/20 text-blue-300 font-semibold"
                : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            <Target className="h-3.5 w-3.5" />
            Accuracy
          </button>
        </div>

        {/* Library selector */}
        <select
          value={compLib}
          onChange={(e) => setCompLib(e.target.value)}
          className="rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
        >
          {COMPARISON_LIBS.map((l) => (
            <option key={l.key} value={l.key}>
              {l.label}
            </option>
          ))}
        </select>

        {/* Extended frames toggle */}
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showExtended}
            onChange={(e) => setShowExtended(e.target.checked)}
            className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
          />
          Show extended frames
        </label>

        {/* Stats badge */}
        {tab === "perf" && perfStats && (
          <span className="ml-auto text-xs text-gray-500">
            Geo. mean:{" "}
            <span className={`${speedupText(parseFloat(perfStats.geo))} px-1.5 py-0.5 rounded ${speedupBg(parseFloat(perfStats.geo))}`}>
              {perfStats.geo}×
            </span>
            {" "}| Wins: {perfStats.wins}/{perfStats.total}
            {" "}| Range: {perfStats.min}× – {perfStats.max}×
          </span>
        )}
      </div>

      {/* Legend */}
      {tab === "perf" ? (
        <div className="flex flex-wrap gap-3 text-[11px] text-gray-500">
          <span>
            Speedup = other / siderust. <strong className="text-emerald-400">&gt; 1×</strong> = siderust faster.
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded bg-emerald-800/80" /> ≥10×
            <span className="inline-block w-3 h-3 rounded bg-emerald-800/60" /> ≥5×
            <span className="inline-block w-3 h-3 rounded bg-emerald-900/60" /> ≥2×
            <span className="inline-block w-3 h-3 rounded bg-emerald-900/30" /> ≥1.2×
            <span className="inline-block w-3 h-3 rounded bg-yellow-900/40" /> ~1×
            <span className="inline-block w-3 h-3 rounded bg-red-900/40" /> &lt;1×
          </span>
        </div>
      ) : (
        <div className="flex flex-wrap gap-3 text-[11px] text-gray-500">
          <span>
            Accuracy = angular error of siderust vs reference (p50 shown). Lower = better.
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded bg-emerald-800/80" /> &lt;0.01 mas
            <span className="inline-block w-3 h-3 rounded bg-emerald-900/60" /> &lt;0.1 mas
            <span className="inline-block w-3 h-3 rounded bg-emerald-900/30" /> &lt;1 mas
            <span className="inline-block w-3 h-3 rounded bg-yellow-900/30" /> &lt;10 mas
            <span className="inline-block w-3 h-3 rounded bg-orange-900/40" /> &lt;100 mas
            <span className="inline-block w-3 h-3 rounded bg-red-900/40" /> ≥100 mas
          </span>
        </div>
      )}

      {!hasAnyData && (
        <div className="rounded-xl border border-yellow-700/40 bg-yellow-950/30 px-5 py-4 text-sm text-yellow-200">
          No data found for coordinate transform experiments in this run.
          Run the experiments (with <code>--perf</code> for performance data) to populate this matrix.
        </div>
      )}

      {/* 2D Grid */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/50 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 bg-gray-900 border border-gray-800/50 px-2 py-2 text-[10px] text-gray-500 font-normal text-left whitespace-nowrap">
                  FROM ↓ &nbsp; TO →
                </th>
                {frames.map((dst) => (
                  <th
                    key={dst}
                    className="border border-gray-800/50 px-1 py-2 text-[10px] text-gray-400 font-medium whitespace-nowrap"
                    style={{ writingMode: "vertical-rl", textOrientation: "mixed", minWidth: 48 }}
                  >
                    {FRAME_SHORT[dst] ?? dst}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {frames.map((src) => (
                <tr key={src}>
                  <td className="sticky left-0 z-10 bg-gray-900 border border-gray-800/50 px-2 py-1 text-[11px] text-gray-400 font-medium whitespace-nowrap">
                    {FRAME_SHORT[src] ?? src}
                  </td>
                  {frames.map((dst) => {
                    if (src === dst) {
                      return (
                        <td
                          key={dst}
                          className="border border-gray-800/50 bg-gray-950/60 text-center text-gray-700 text-[10px] p-1"
                        >
                          —
                        </td>
                      );
                    }
                    const cellKey = `${src}→${dst}`;
                    if (tab === "perf") {
                      const cell = activeGrid?.[cellKey] as PerfMatrixCell | undefined;
                      return <PerfCell key={dst} cell={cell} />;
                    } else {
                      const cell = activeGrid?.[cellKey] as AccuracyMatrixCell | undefined;
                      return <AccCell key={dst} cell={cell} />;
                    }
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Experiment reference table */}
      <details className="rounded-xl border border-gray-800 bg-gray-900/50">
        <summary className="px-5 py-3 text-sm text-gray-400 cursor-pointer hover:text-gray-300">
          Experiment → Cell mapping reference
        </summary>
        <div className="px-5 pb-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-1 text-xs text-gray-500">
          {Object.entries(data.experiment_map).map(([exp, { from, to }]) => (
            <div key={exp} className="flex gap-2">
              <span className="text-gray-400 font-mono">{EXPERIMENT_LABELS[exp] ?? exp}</span>
              <span className="text-gray-600">
                {FRAME_SHORT[from] ?? from} → {FRAME_SHORT[to] ?? to}
              </span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
