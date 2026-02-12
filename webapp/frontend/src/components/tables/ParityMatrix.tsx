import type { ExperimentResult } from "../../api/types";

interface Props {
  experiments: Record<string, ExperimentResult[]>;
}

export default function ParityMatrix({ experiments }: Props) {
  // Collect all libraries across experiments
  const libSet = new Set<string>(["erfa"]); // reference always included
  for (const results of Object.values(experiments)) {
    for (const r of results) libSet.add(r.candidate_library);
  }
  const libs = Array.from(libSet).sort();

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase text-gray-400">
            <th className="px-4 py-3 min-w-[160px]">Experiment</th>
            {libs.map((lib) => (
              <th key={lib} className="px-4 py-3 min-w-[200px]">
                {lib}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(experiments).map(([exp, results], idx) => {
            const models =
              results[0]?.alignment?.models ?? ({} as Record<string, string>);
            return (
              <tr
                key={exp}
                className={`border-b border-gray-800/50 ${
                  idx % 2 === 0 ? "bg-gray-900/30" : ""
                }`}
              >
                <td className="px-4 py-2.5 font-medium text-gray-200">
                  {exp}
                </td>
                {libs.map((lib) => (
                  <td key={lib} className="px-4 py-2.5 text-gray-400 max-w-xs">
                    <span className="line-clamp-3">{models[lib] ?? "\u2014"}</span>
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
