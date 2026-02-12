import { Link } from "react-router-dom";
import type { ExperimentResult } from "../../api/types";
import { LIBRARY_COLORS } from "../../api/types";

interface Props {
  runId: string;
  experiments: Record<string, ExperimentResult[]>;
}

function fmt(v: unknown, prec = 4): string {
  if (v == null) return "\u2014";
  const n = Number(v);
  if (isNaN(n)) return "\u2014";
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
  return n.toFixed(prec);
}

/** Extract a primary error metric from accuracy depending on experiment type. */
function primaryError(r: ExperimentResult): { p99: string; max: string; unit: string } {
  const acc = r.accuracy as Record<string, Record<string, number | null> | undefined>;

  // BPN
  if (acc.angular_error_mas) {
    return {
      p99: fmt(acc.angular_error_mas.p99, 2),
      max: fmt(acc.angular_error_mas.max, 2),
      unit: "mas",
    };
  }
  // GMST
  if (acc.gmst_error_arcsec) {
    return {
      p99: fmt(acc.gmst_error_arcsec.p99, 6),
      max: fmt(acc.gmst_error_arcsec.max, 6),
      unit: "arcsec",
    };
  }
  // Angular (equ_ecl, equ_horizontal, solar, lunar)
  if (acc.angular_sep_arcsec) {
    return {
      p99: fmt(acc.angular_sep_arcsec.p99),
      max: fmt(acc.angular_sep_arcsec.max),
      unit: "arcsec",
    };
  }
  // Kepler
  if (acc.E_error_rad) {
    return {
      p99: fmt(acc.E_error_rad.p99),
      max: fmt(acc.E_error_rad.max),
      unit: "rad",
    };
  }
  return { p99: "\u2014", max: "\u2014", unit: "" };
}

export default function SummaryTable({ runId, experiments }: Props) {
  const rows: {
    experiment: string;
    library: string;
    p99: string;
    max: string;
    unit: string;
    nsOp: string;
  }[] = [];

  for (const [exp, results] of Object.entries(experiments)) {
    for (const r of results) {
      const err = primaryError(r);
      const perf = r.performance as Record<string, unknown>;
      rows.push({
        experiment: exp,
        library: r.candidate_library,
        ...err,
        nsOp: perf?.per_op_ns != null ? fmt(perf.per_op_ns, 0) : "\u2014",
      });
    }
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase text-gray-400">
            <th className="px-4 py-3">Experiment</th>
            <th className="px-4 py-3">Library</th>
            <th className="px-4 py-3">p99 Error</th>
            <th className="px-4 py-3">Max Error</th>
            <th className="px-4 py-3">Unit</th>
            <th className="px-4 py-3">ns/op</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={`${r.experiment}-${r.library}`}
              className={`border-b border-gray-800/50 hover:bg-gray-800/40 ${
                i % 2 === 0 ? "bg-gray-900/30" : ""
              }`}
            >
              <td className="px-4 py-2.5">
                <Link
                  to={`/runs/${runId}/experiments/${r.experiment}`}
                  className="text-blue-400 hover:underline"
                >
                  {r.experiment}
                </Link>
              </td>
              <td className="px-4 py-2.5">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full mr-2"
                  style={{
                    backgroundColor: LIBRARY_COLORS[r.library] ?? "#6b7280",
                  }}
                />
                {r.library}
              </td>
              <td className="px-4 py-2.5 font-mono">{r.p99}</td>
              <td className="px-4 py-2.5 font-mono">{r.max}</td>
              <td className="px-4 py-2.5 text-gray-500">{r.unit}</td>
              <td className="px-4 py-2.5 font-mono">{r.nsOp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
