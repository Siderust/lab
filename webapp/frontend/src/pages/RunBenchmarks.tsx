import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { startBenchmark, subscribeBenchmarkLogs } from "../api/client";
import type { BenchmarkRequest } from "../api/types";
import Header from "../components/layout/Header";
import BenchmarkForm from "../components/benchmark/BenchmarkForm";
import LogStream from "../components/benchmark/LogStream";

export default function RunBenchmarks() {
  const navigate = useNavigate();
  const [, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [lines, setLines] = useState<string[]>([]);

  const handleSubmit = useCallback(
    async (request: BenchmarkRequest) => {
      setLines([]);
      setStatus("starting");

      try {
        const job = await startBenchmark(request);
        setJobId(job.job_id);
        setStatus("running");

        subscribeBenchmarkLogs(
          job.job_id,
          (line) => setLines((prev) => [...prev, line]),
          (finalStatus) => {
            setStatus(finalStatus);
            if (finalStatus === "completed") {
              setLines((prev) => [
                ...prev,
                "",
                "--- Benchmark completed. Reloading runs... ---",
              ]);
              // Navigate to runs list after a short delay
              setTimeout(() => navigate("/"), 2000);
            }
          }
        );
      } catch (err) {
        setStatus("error");
        setLines((prev) => [
          ...prev,
          `Error: ${err instanceof Error ? err.message : String(err)}`,
        ]);
      }
    },
    [navigate]
  );

  const isRunning = status === "running" || status === "starting";

  return (
    <div>
      <Header
        title="Run Benchmarks"
        subtitle="Select experiments, set parameters, and launch. Output streams live below."
      />

      <div className="space-y-6">
        <BenchmarkForm onSubmit={handleSubmit} disabled={isRunning} />

        {(lines.length > 0 || isRunning) && (
          <LogStream lines={lines} status={status} />
        )}

        {status === "completed" && (
          <p className="text-green-400 text-sm">
            Benchmark completed successfully. Redirecting to runs list...
          </p>
        )}
        {status === "failed" && (
          <p className="text-red-400 text-sm">
            Benchmark failed. Check the output above for details.
          </p>
        )}
      </div>
    </div>
  );
}
