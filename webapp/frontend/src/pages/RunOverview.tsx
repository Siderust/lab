import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { fetchRun } from "../api/client";
import type { ExperimentResult } from "../api/types";
import Header from "../components/layout/Header";
import SummaryTable from "../components/tables/SummaryTable";
import ParityMatrix from "../components/tables/ParityMatrix";
import ParetoPlot from "../components/charts/ParetoPlot";
import AlertCard from "../components/cards/AlertCard";

/** Group experiments by "family" for Pareto plots */
const FAMILIES: Record<string, string[]> = {
  Frames: ["frame_rotation_bpn", "equ_ecl", "equ_horizontal"],
  Time: ["gmst_era"],
  Ephemerides: ["solar_position", "lunar_position"],
  Orbits: ["kepler_solver"],
};

/**
 * Extract the primary p99 error and ns/op from an ExperimentResult
 * so we can plot Pareto points.
 */
function paretoPoint(r: ExperimentResult) {
  const acc = r.accuracy as Record<string, Record<string, number | null> | undefined>;
  const perf = r.performance as Record<string, unknown>;

  let error: number | null = null;
  if (acc.angular_error_mas?.p99 != null) error = acc.angular_error_mas.p99;
  else if (acc.gmst_error_arcsec?.p99 != null) error = acc.gmst_error_arcsec.p99;
  else if (acc.angular_sep_arcsec?.p99 != null) error = acc.angular_sep_arcsec.p99;
  else if (acc.E_error_rad?.p99 != null) error = acc.E_error_rad.p99;

  const latency = (perf?.per_op_ns as number) ?? null;

  if (error == null || latency == null || error === 0) return null;
  return { library: r.candidate_library, error, latency };
}

export default function RunOverview() {
  const { runId } = useParams<{ runId: string }>();
  const { data: run, isLoading, error } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => fetchRun(runId!),
    enabled: !!runId,
  });

  if (isLoading) return <p className="text-gray-500">Loading run...</p>;
  if (error || !run)
    return <p className="text-red-400">Failed to load run {runId}.</p>;

  // Compute alerts
  const alerts: { level: "info" | "warn" | "error"; message: string }[] = [];
  for (const [exp, results] of Object.entries(run.experiments)) {
    for (const r of results) {
      const acc = r.accuracy as Record<string, unknown>;
      const nan = (acc.nan_count as number) ?? 0;
      const inf = (acc.inf_count as number) ?? 0;
      if (nan > 0) alerts.push({ level: "warn", message: `${exp}/${r.candidate_library}: ${nan} NaN results` });
      if (inf > 0) alerts.push({ level: "warn", message: `${exp}/${r.candidate_library}: ${inf} Inf results` });
      const perf = r.performance as Record<string, unknown>;
      const hasPerf = perf && Object.keys(perf).length > 0 && perf.per_op_ns != null;
      if (!hasPerf) alerts.push({ level: "info", message: `${exp}/${r.candidate_library}: no performance data` });
    }
  }
  // Deduplicate "no performance data" if all experiments lack it
  const perfMissing = alerts.filter((a) => a.message.includes("no performance data"));
  const otherAlerts = alerts.filter((a) => !a.message.includes("no performance data"));
  const deduped =
    perfMissing.length === Object.values(run.experiments).flat().length
      ? [
          ...otherAlerts,
          { level: "info" as const, message: "No performance data in this run." },
        ]
      : alerts;

  return (
    <div>
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

      {/* Alerts */}
      {deduped.length > 0 && (
        <div className="space-y-2 mb-6">
          {deduped.slice(0, 5).map((a, i) => (
            <AlertCard key={i} level={a.level} message={a.message} />
          ))}
          {deduped.length > 5 && (
            <p className="text-xs text-gray-500">
              ... and {deduped.length - 5} more alerts
            </p>
          )}
        </div>
      )}

      {/* Summary Table */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-white mb-3">
          Summary (Experiments &times; Libraries)
        </h2>
        <SummaryTable runId={run.id} experiments={run.experiments} />
      </section>

      {/* Pareto Plots per Family */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-white mb-3">
          Pareto: Accuracy vs Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(FAMILIES).map(([family, expNames]) => {
            const points = expNames.flatMap((exp) =>
              (run.experiments[exp] ?? [])
                .map(paretoPoint)
                .filter((p): p is NonNullable<typeof p> => p !== null)
            );
            return (
              <ParetoPlot
                key={family}
                points={points}
                title={`${family}: Error vs Latency`}
              />
            );
          })}
        </div>
      </section>

      {/* Parity Matrix */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-white mb-3">
          Feature / Model Parity Matrix
        </h2>
        <ParityMatrix experiments={run.experiments} />
      </section>

      {/* Git SHAs */}
      {Object.keys(run.git_shas).length > 0 && (
        <section className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
            Git SHAs
          </h3>
          <div className="flex flex-wrap gap-4">
            {Object.entries(run.git_shas).map(([repo, sha]) => (
              <div key={repo} className="text-xs">
                <span className="text-gray-400">{repo}:</span>{" "}
                <span className="font-mono text-gray-300">{sha}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
