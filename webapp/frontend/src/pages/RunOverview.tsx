/**
 * RunOverview — Primary dashboard for a single benchmark run.
 *
 * Sections:
 *  1. Overall ranking summary cards
 *  2. Performance comparison (grouped bar chart, log-scale)
 *  3. Accuracy comparison (grouped bar chart, log-scale)
 *  4. Detailed results table with rankings
 *  5. Cross-comparison matrix (pairwise wins + heatmaps)
 *  6. Run metadata
 */

import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Trophy, Zap, Target, Scale } from "lucide-react";
import { fetchRun } from "../api/client";
import { analyzeRun, libColor } from "../utils/analytics";
import Header from "../components/layout/Header";
import PerformanceChart from "../components/charts/PerformanceChart";
import AccuracyChart from "../components/charts/AccuracyChart";
import ComparisonMatrix from "../components/charts/ComparisonMatrix";
import SummaryTable from "../components/tables/SummaryTable";

export default function RunOverview() {
  const { runId } = useParams<{ runId: string }>();
  const {
    data: run,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => fetchRun(runId!),
    enabled: !!runId,
  });

  if (isLoading) return <p className="text-gray-500">Loading run...</p>;
  if (error || !run)
    return <p className="text-red-400">Failed to load run {runId}.</p>;

  const analytics = analyzeRun(run);

  // Count data quality issues
  const alerts: string[] = [];
  for (const [exp, results] of Object.entries(run.experiments)) {
    for (const r of results) {
      const acc = r.accuracy as Record<string, unknown>;
      const nan = (acc.nan_count as number) ?? 0;
      const inf = (acc.inf_count as number) ?? 0;
      if (nan > 0)
        alerts.push(
          `${exp}/${r.candidate_library}: ${nan} NaN results`,
        );
      if (inf > 0)
        alerts.push(
          `${exp}/${r.candidate_library}: ${inf} Inf results`,
        );
    }
  }

  // Count experiments with performance data
  const expWithPerf = analytics.experiments.filter(
    (e) => e.performance.filter((p) => !p.isReference).length > 0,
  ).length;

  return (
    <div className="space-y-8">
      <Header
        title={`Run ${run.id}`}
        subtitle={`${run.timestamp ?? ""} \u2014 ${run.machine ?? "unknown machine"}`}
        actions={
          <Link
            to="/"
            className="flex items-center gap-2 rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700"
          >
            <ArrowLeft className="h-4 w-4" />
            All Runs
          </Link>
        }
      />

      {/* ─── Data quality alerts ──────────────────────────────── */}
      {alerts.length > 0 && (
        <div className="rounded-xl border border-yellow-700/40 bg-yellow-950/30 px-4 py-3">
          <p className="text-xs font-medium text-yellow-300 mb-1">
            Data Quality Warnings
          </p>
          <ul className="text-xs text-yellow-200/70 space-y-0.5">
            {alerts.slice(0, 5).map((a, i) => (
              <li key={i}>{a}</li>
            ))}
            {alerts.length > 5 && (
              <li className="text-yellow-400/50">
                ...and {alerts.length - 5} more
              </li>
            )}
          </ul>
        </div>
      )}

      {/* ─── Overall ranking cards ────────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Trophy className="h-5 w-5 text-yellow-400" />
          Overall Rankings
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <RankingCard
            icon={<Zap className="h-5 w-5 text-yellow-400" />}
            title="Fastest Library"
            library={analytics.overallFastest}
            subtitle={`Wins ${countWins(analytics, "fastest")} of ${analytics.experiments.length} experiments`}
            accentColor="yellow"
          />
          <RankingCard
            icon={<Target className="h-5 w-5 text-emerald-400" />}
            title="Most Accurate Library"
            library={analytics.overallMostAccurate}
            subtitle={`Wins ${countWins(analytics, "mostAccurate")} of ${analytics.experiments.length} experiments`}
            accentColor="emerald"
          />
          <RankingCard
            icon={<Scale className="h-5 w-5 text-blue-400" />}
            title="Best Tradeoff"
            library={analytics.overallBestTradeoff}
            subtitle="Most combined performance + accuracy wins"
            accentColor="blue"
          />
        </div>
      </section>

      {/* ─── Performance comparison ───────────────────────────── */}
      {expWithPerf > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-400" />
            Performance Comparison
          </h2>
          <PerformanceChart experiments={analytics.experiments} />
        </section>
      )}

      {/* ─── Accuracy comparison ──────────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Target className="h-5 w-5 text-emerald-400" />
          Accuracy Comparison
        </h2>
        <AccuracyChart experiments={analytics.experiments} />
      </section>

      {/* ─── Detailed results table ───────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">
          Detailed Results
        </h2>
        <SummaryTable runId={run.id} analytics={analytics} />
      </section>

      {/* ─── Cross-comparison matrix ──────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">
          Cross-Comparison Matrix
        </h2>
        <ComparisonMatrix analytics={analytics} />
      </section>

      {/* ─── Benchmark Configuration ──────────────────────────── */}
      {(() => {
        const firstResult = Object.values(run.experiments)[0]?.[0];
        const config = firstResult?.benchmark_config as Record<string, unknown> | undefined;
        const inputs = firstResult?.inputs as Record<string, unknown> | undefined;
        const meta = firstResult?.run_metadata;

        return (config || inputs || meta) ? (
          <section className="rounded-xl border border-gray-800 bg-gray-900/40 p-4 space-y-4">
            <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
              Run Configuration & Environment
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {inputs?.count != null && (
                <div className="text-xs">
                  <span className="text-gray-500">Test Cases (N):</span>{" "}
                  <span className="font-mono text-gray-300">{String(inputs.count)}</span>
                </div>
              )}
              {inputs?.seed != null && (
                <div className="text-xs">
                  <span className="text-gray-500">Seed:</span>{" "}
                  <span className="font-mono text-gray-300">{String(inputs.seed)}</span>
                </div>
              )}
              {config?.perf_rounds != null && (
                <div className="text-xs">
                  <span className="text-gray-500">Perf Rounds:</span>{" "}
                  <span className="font-mono text-gray-300">{String(config.perf_rounds)}</span>
                </div>
              )}
              {inputs?.dataset_fingerprint != null && (
                <div className="text-xs">
                  <span className="text-gray-500">Dataset ID:</span>{" "}
                  <span className="font-mono text-gray-300">{String(inputs.dataset_fingerprint)}</span>
                </div>
              )}
            </div>

            {/* Environment info */}
            {meta && (
              <div className="border-t border-gray-800 pt-3 grid grid-cols-2 sm:grid-cols-3 gap-3">
                {meta.cpu_model && (
                  <div className="text-xs">
                    <span className="text-gray-500">CPU:</span>{" "}
                    <span className="text-gray-300">{meta.cpu_model}</span>
                  </div>
                )}
                {meta.cpu_count != null && (
                  <div className="text-xs">
                    <span className="text-gray-500">Cores:</span>{" "}
                    <span className="text-gray-300">{meta.cpu_count}</span>
                  </div>
                )}
                {meta.os && (
                  <div className="text-xs">
                    <span className="text-gray-500">OS:</span>{" "}
                    <span className="text-gray-300">{meta.os}</span>
                  </div>
                )}
                {meta.git_branch && (
                  <div className="text-xs">
                    <span className="text-gray-500">Branch:</span>{" "}
                    <span className="font-mono text-gray-300">{meta.git_branch}</span>
                  </div>
                )}
                {meta.toolchain && Object.entries(meta.toolchain).map(([tool, version]) => (
                  <div key={tool} className="text-xs">
                    <span className="text-gray-500">{tool}:</span>{" "}
                    <span className="text-gray-300">{version}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Git SHAs */}
            {Object.keys(run.git_shas).length > 0 && (
              <div className="border-t border-gray-800 pt-3">
                <p className="text-xs text-gray-500 mb-1.5">Git SHAs:</p>
                <div className="flex flex-wrap gap-4">
                  {Object.entries(run.git_shas).map(([repo, sha]) => (
                    <div key={repo} className="text-xs">
                      <span className="text-gray-400">{repo}:</span>{" "}
                      <span className="font-mono text-gray-300">{sha}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        ) : null;
      })()}
    </div>
  );
}

// ─── Helper components ───────────────────────────────────────────────

function RankingCard({
  icon,
  title,
  library,
  subtitle,
  accentColor,
}: {
  icon: React.ReactNode;
  title: string;
  library: string | null;
  subtitle: string;
  accentColor: "yellow" | "emerald" | "blue";
}) {
  const borderColors = {
    yellow: "border-yellow-600/40",
    emerald: "border-emerald-600/40",
    blue: "border-blue-600/40",
  };

  return (
    <div
      className={`rounded-xl border bg-gray-900/60 px-5 py-4 ${borderColors[accentColor]}`}
    >
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {title}
        </p>
      </div>
      <p className="text-xl font-bold text-white">
        {library ? (
          <span className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ backgroundColor: libColor(library) }}
            />
            {library}
          </span>
        ) : (
          <span className="text-gray-500">N/A</span>
        )}
      </p>
      <p className="mt-1 text-xs text-gray-500">{subtitle}</p>
    </div>
  );
}

function countWins(
  analytics: ReturnType<typeof analyzeRun>,
  field: "fastest" | "mostAccurate",
): number {
  const target =
    field === "fastest"
      ? analytics.overallFastest
      : analytics.overallMostAccurate;
  if (!target) return 0;
  return analytics.experiments.filter((e) => e[field] === target).length;
}
