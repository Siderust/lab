import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Eye, GitCompareArrows, Download, RefreshCw } from "lucide-react";
import { fetchRuns, reloadRuns } from "../api/client";
import Header from "../components/layout/Header";

export default function RunsList() {
  const { data: runs, isLoading, refetch } = useQuery({
    queryKey: ["runs"],
    queryFn: fetchRuns,
  });

  const handleReload = async () => {
    await reloadRuns();
    refetch();
  };

  return (
    <div>
      <Header
        title="Benchmark Runs"
        subtitle="Browse past runs, view dashboards, compare, and download artifacts."
        actions={
          <button
            onClick={handleReload}
            className="flex items-center gap-2 rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Reload
          </button>
        }
      />

      {isLoading && (
        <p className="text-gray-500 text-sm">Loading runs...</p>
      )}

      {runs && runs.length === 0 && (
        <div className="text-center py-20 text-gray-500">
          <p className="text-lg font-medium mb-2">No runs found</p>
          <p className="text-sm">
            Run benchmarks first, or make sure <code>results/</code> is
            populated.
          </p>
        </div>
      )}

      {runs && runs.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase text-gray-400">
                <th className="px-4 py-3">Run ID</th>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Machine</th>
                <th className="px-4 py-3">Experiments</th>
                <th className="px-4 py-3">Libraries</th>
                <th className="px-4 py-3">Results</th>
                <th className="px-4 py-3">Reports</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run, i) => (
                <tr
                  key={run.id}
                  className={`border-b border-gray-800/50 hover:bg-gray-800/40 ${
                    i % 2 === 0 ? "bg-gray-900/30" : ""
                  }`}
                >
                  <td className="px-4 py-3 font-mono font-medium text-white">
                    {run.id}
                  </td>
                  <td className="px-4 py-3 text-gray-400">
                    {run.timestamp ?? "\u2014"}
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs max-w-xs truncate">
                    {run.machine ?? "\u2014"}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-gray-300">
                      {run.experiments.length}
                    </span>
                    <span className="text-gray-600 ml-1 text-xs">
                      ({run.experiments.join(", ")})
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-400">
                    {run.libraries.join(", ")}
                  </td>
                  <td className="px-4 py-3 text-gray-400">
                    {run.result_count}
                  </td>
                  <td className="px-4 py-3">
                    {run.has_reports ? (
                      <span className="text-green-400 text-xs">Yes</span>
                    ) : (
                      <span className="text-gray-600 text-xs">No</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Link
                        to={`/runs/${run.id}`}
                        className="rounded-md p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
                        title="View dashboard"
                      >
                        <Eye className="h-4 w-4" />
                      </Link>
                      <Link
                        to={`/compare?run_a=${run.id}`}
                        className="rounded-md p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
                        title="Compare"
                      >
                        <GitCompareArrows className="h-4 w-4" />
                      </Link>
                      <a
                        href={`/api/runs/${run.id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-md p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
                        title="Download JSON"
                      >
                        <Download className="h-4 w-4" />
                      </a>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Git SHA details */}
      {runs && runs.length > 0 && runs[0].git_shas && (
        <div className="mt-6 rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <h3 className="text-xs font-medium uppercase text-gray-400 mb-2">
            Latest Run Git SHAs
          </h3>
          <div className="flex flex-wrap gap-4">
            {Object.entries(runs[0].git_shas).map(([repo, sha]) => (
              <div key={repo} className="text-xs">
                <span className="text-gray-400">{repo}:</span>{" "}
                <span className="font-mono text-gray-300">{sha}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
