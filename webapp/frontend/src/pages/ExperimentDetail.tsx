import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { fetchExperiment } from "../api/client";
import type { ExperimentResult, PercentileStats } from "../api/types";
import Header from "../components/layout/Header";
import MetricCard from "../components/cards/MetricCard";
import CdfPlot from "../components/charts/CdfPlot";
import ErrorVsEpoch from "../components/charts/ErrorVsEpoch";
import BarChart from "../components/charts/BarChart";
import OutliersTable from "../components/tables/OutliersTable";
import type { OutlierRow } from "../components/tables/OutliersTable";

const TABS = ["Overview", "Accuracy", "Performance", "Outliers", "Assumptions"] as const;
type Tab = (typeof TABS)[number];

// ── helpers ──────────────────────────────────────────────────────────

type AccDict = Record<string, Record<string, number | null> | undefined>;

function stats(acc: AccDict, key: string): PercentileStats | null {
  const s = acc[key] as PercentileStats | undefined;
  return s ?? null;
}

/** Given the accuracy dict, figure out the primary error metric info. */
function primaryMetric(acc: AccDict): {
  key: string;
  label: string;
  unit: string;
} {
  if (acc.angular_error_mas) return { key: "angular_error_mas", label: "Angular error", unit: "mas" };
  if (acc.gmst_error_arcsec) return { key: "gmst_error_arcsec", label: "GMST error", unit: "arcsec" };
  if (acc.angular_sep_arcsec) return { key: "angular_sep_arcsec", label: "Angular sep", unit: "arcsec" };
  if (acc.E_error_rad) return { key: "E_error_rad", label: "E error", unit: "rad" };
  return { key: "", label: "Error", unit: "" };
}

// ── component ────────────────────────────────────────────────────────

export default function ExperimentDetail() {
  const { runId, experiment } = useParams<{
    runId: string;
    experiment: string;
  }>();
  const [tab, setTab] = useState<Tab>("Overview");

  const { data: results, isLoading, error } = useQuery({
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
        title={experiment!}
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

      {/* Tab content */}
      {tab === "Overview" && <OverviewTab results={results} />}
      {tab === "Accuracy" && <AccuracyTab results={results} />}
      {tab === "Performance" && <PerformanceTab results={results} />}
      {tab === "Outliers" && <OutliersTab results={results} />}
      {tab === "Assumptions" && <AssumptionsTab results={results} />}
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────────────

function OverviewTab({ results }: { results: ExperimentResult[] }) {
  return (
    <div className="space-y-6">
      {results.map((r) => {
        const acc = r.accuracy as AccDict;
        const pm = primaryMetric(acc);
        const s = stats(acc, pm.key);
        const nan = (acc.nan_count as unknown as number) ?? 0;
        const inf = (acc.inf_count as unknown as number) ?? 0;

        return (
          <div key={r.candidate_library}>
            <h3 className="text-sm font-semibold text-gray-300 mb-3">
              {r.candidate_library} vs {r.reference_library}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <MetricCard label={`${pm.label} p50`} value={s?.p50} unit={pm.unit} />
              <MetricCard label={`${pm.label} p99`} value={s?.p99} unit={pm.unit} />
              <MetricCard label={`${pm.label} max`} value={s?.max} unit={pm.unit} />
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
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4 mt-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-1">
            Parity Note
          </h3>
          <p className="text-sm text-gray-300">{results[0].alignment.note}</p>
        </div>
      )}
    </div>
  );
}

// ── Accuracy Tab ─────────────────────────────────────────────────────

function AccuracyTab({ results }: { results: ExperimentResult[] }) {
  // Build CDF and error-vs-epoch series from the worst_cases data and summary stats.
  // The worst_cases in BPN give us (jd_tt, angular_error_mas).
  // For other experiments we only have aggregate stats — we render what we have.

  const pm = primaryMetric(results[0]?.accuracy as AccDict);

  // CDF: we simulate from percentile stats (p50, p90, p95, p99, max)
  // since per-case data isn't stored yet.
  const cdfSeries = results
    .map((r) => {
      const s = stats(r.accuracy as AccDict, pm.key);
      if (!s) return null;
      // Build a pseudo-CDF from known percentiles
      const values: number[] = [];
      if (s.min != null) values.push(s.min);
      if (s.p50 != null) values.push(s.p50);
      if (s.p90 != null) values.push(s.p90);
      if (s.p95 != null) values.push(s.p95);
      if (s.p99 != null) values.push(s.p99);
      if (s.max != null) values.push(s.max);
      return { library: r.candidate_library, values };
    })
    .filter((s): s is NonNullable<typeof s> => s !== null);

  // Error vs epoch from worst_cases (BPN only has this)
  const epochSeries = results
    .map((r) => {
      const acc = r.accuracy as Record<string, unknown>;
      const wc = acc.worst_cases as
        | { jd_tt: number; angular_error_mas: number }[]
        | undefined;
      if (!wc || wc.length === 0) return null;
      return {
        library: r.candidate_library,
        epochs: wc.map((c) => c.jd_tt),
        errors: wc.map((c) => c.angular_error_mas),
      };
    })
    .filter((s): s is NonNullable<typeof s> => s !== null);

  // Bias stats
  const biasSeries = results.map((r) => {
    const acc = r.accuracy as AccDict;
    return {
      library: r.candidate_library,
      ra_bias: stats(acc, "signed_ra_error_arcsec"),
      dec_bias: stats(acc, "signed_dec_error_arcsec"),
    };
  });

  return (
    <div className="space-y-6">
      {cdfSeries.length > 0 && (
        <CdfPlot
          series={cdfSeries}
          xLabel={`${pm.label} (${pm.unit})`}
          title={`CDF of ${pm.label}`}
        />
      )}

      {epochSeries.length > 0 && (
        <ErrorVsEpoch
          series={epochSeries}
          yLabel={`${pm.label} (${pm.unit})`}
          title="Error vs Epoch (Worst Cases)"
        />
      )}

      {/* Bias summary cards */}
      {biasSeries.some((b) => b.ra_bias || b.dec_bias) && (
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Signed Bias
          </h3>
          {biasSeries.map((b) =>
            b.ra_bias || b.dec_bias ? (
              <div
                key={b.library}
                className="grid grid-cols-4 gap-3 mb-3"
              >
                <MetricCard
                  label={`${b.library} RA bias`}
                  value={b.ra_bias?.mean}
                  unit="arcsec"
                />
                <MetricCard
                  label={`${b.library} Dec bias`}
                  value={b.dec_bias?.mean}
                  unit="arcsec"
                />
              </div>
            ) : null
          )}
        </div>
      )}

      {cdfSeries.length === 0 && epochSeries.length === 0 && (
        <p className="text-sm text-gray-500 italic">
          No per-case accuracy data available for plotting. Only aggregate
          statistics are shown in the Overview tab.
        </p>
      )}
    </div>
  );
}

// ── Performance Tab ──────────────────────────────────────────────────

function PerformanceTab({ results }: { results: ExperimentResult[] }) {
  const bars = results
    .map((r) => {
      const perf = r.performance as Record<string, unknown>;
      const ns = perf?.per_op_ns as number | undefined;
      if (ns == null) return null;
      return { library: r.candidate_library, value: ns };
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);

  const throughputBars = results
    .map((r) => {
      const perf = r.performance as Record<string, unknown>;
      const ops = perf?.throughput_ops_s as number | undefined;
      if (ops == null) return null;
      return { library: r.candidate_library, value: ops };
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);

  return (
    <div className="space-y-6">
      <BarChart bars={bars} yLabel="ns/op" title="Latency (ns/op)" />
      {throughputBars.length > 0 && (
        <BarChart
          bars={throughputBars}
          yLabel="ops/s"
          title="Throughput (ops/s)"
        />
      )}

      {/* Reference performance */}
      {results[0]?.reference_performance && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
            Reference ({results[0].reference_library}) Performance
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="ns/op"
              value={
                (results[0].reference_performance as Record<string, unknown>)
                  .per_op_ns as number
              }
              unit="ns"
            />
            <MetricCard
              label="throughput"
              value={
                (results[0].reference_performance as Record<string, unknown>)
                  .throughput_ops_s as number
              }
              unit="ops/s"
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Outliers Tab ─────────────────────────────────────────────────────

function OutliersTab({ results }: { results: ExperimentResult[] }) {
  const rows: OutlierRow[] = [];

  for (const r of results) {
    const acc = r.accuracy as Record<string, unknown>;
    const wc = acc.worst_cases as
      | { jd_tt: number; angular_error_mas: number }[]
      | undefined;
    if (wc) {
      for (const c of wc) {
        rows.push({
          library: r.candidate_library,
          jd_tt: c.jd_tt,
          error: c.angular_error_mas,
          unit: "mas",
        });
      }
    }
  }

  // Sort by descending error
  rows.sort((a, b) => (b.error ?? 0) - (a.error ?? 0));

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Worst-N Cases
      </h3>
      <OutliersTable rows={rows} />
    </div>
  );
}

// ── Assumptions Tab ──────────────────────────────────────────────────

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
      {/* Structured display */}
      {Object.entries(alignment).map(([key, value]) => {
        if (key === "models") {
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
                  )
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
                  )
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
