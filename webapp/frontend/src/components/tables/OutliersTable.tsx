interface OutlierRow {
  library: string;
  jd_tt: number | null;
  error: number | null;
  unit: string;
}

interface Props {
  rows: OutlierRow[];
}

function fmt(v: number | null): string {
  if (v == null) return "\u2014";
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
  return v.toFixed(4);
}

export default function OutliersTable({ rows }: Props) {
  if (rows.length === 0) {
    return (
      <p className="text-sm text-gray-500 italic">
        No outlier data available for this experiment.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase text-gray-400">
            <th className="px-4 py-3">#</th>
            <th className="px-4 py-3">Library</th>
            <th className="px-4 py-3">JD(TT)</th>
            <th className="px-4 py-3">Error</th>
            <th className="px-4 py-3">Unit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className={`border-b border-gray-800/50 ${
                i % 2 === 0 ? "bg-gray-900/30" : ""
              }`}
            >
              <td className="px-4 py-2.5 text-gray-500">{i + 1}</td>
              <td className="px-4 py-2.5">{r.library}</td>
              <td className="px-4 py-2.5 font-mono">
                {r.jd_tt?.toFixed(6) ?? "\u2014"}
              </td>
              <td className="px-4 py-2.5 font-mono">{fmt(r.error)}</td>
              <td className="px-4 py-2.5 text-gray-500">{r.unit}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export type { OutlierRow };
