import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { startBenchmark, subscribeBenchmarkLogs, fetchJobs, fetchJobLogs } from "../api/client";
import type { BenchmarkRequest, BenchmarkStatus } from "../api/types";
import Header from "../components/layout/Header";
import BenchmarkForm from "../components/benchmark/BenchmarkForm";
import LogStream from "../components/benchmark/LogStream";

export default function RunBenchmarks() {
  const navigate = useNavigate();
  const [, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");

  // View-logs state for past jobs
  const [viewingJobId, setViewingJobId] = useState<string | null>(null);
  const [viewingLines, setViewingLines] = useState<string[]>([]);
  const [viewingStatus, setViewingStatus] = useState<string>("idle");

  // Past jobs list
  const { data: jobs, refetch: refetchJobs } = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: status === "running" ? 5000 : false,
  });

  // Refetch jobs when a run finishes
  useEffect(() => {
    if (status === "completed" || status === "failed") {
      refetchJobs();
    }
  }, [status, refetchJobs]);

  const handleSubmit = useCallback(
    async (request: BenchmarkRequest) => {
      setStatus("starting");

      try {
        const job = await startBenchmark(request);
        setJobId(job.job_id);
        setStatus("running");

        subscribeBenchmarkLogs(
          job.job_id,
          () => {},
          (finalStatus) => {
            setStatus(finalStatus);
            if (finalStatus === "completed") {
              // Navigate to runs list after a short delay
              setTimeout(() => navigate("/"), 2000);
            }
          }
        );
      } catch (err) {
        setStatus("error");
        console.error("Failed to start benchmark run:", err);
      }
    },
    [navigate]
  );

  const handleViewLogs = useCallback(async (jobId: string) => {
    setViewingJobId(jobId);
    setViewingStatus("loading");
    try {
      const text = await fetchJobLogs(jobId);
      setViewingLines(text.split("\n"));
      setViewingStatus("loaded");
    } catch {
      setViewingLines(["Failed to load logs."]);
      setViewingStatus("error");
    }
  }, []);

  const isRunning = status === "running" || status === "starting";

  return (
    <div>
      <Header
        title="Run Benchmarks"
        subtitle="Select experiments, set parameters, and launch."
      />

      <div className="space-y-6">
        <BenchmarkForm onSubmit={handleSubmit} disabled={isRunning} />

        {status === "completed" && (
          <p className="text-green-400 text-sm">
            Benchmark completed successfully. Redirecting to runs list...
          </p>
        )}
        {status === "failed" && (
          <p className="text-red-400 text-sm">
            Benchmark failed. Check Job History and open logs for details.
          </p>
        )}
        {status === "closed" && (
          <p className="text-yellow-400 text-sm">
            Connection lost. The benchmark may still be running in the background.
            Check the job history below once it finishes.
          </p>
        )}
        {status === "error" && (
          <p className="text-red-400 text-sm">
            WebSocket error. The benchmark may still be running.
            Check the job history below.
          </p>
        )}

        {/* Job history */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60">
          <div className="border-b border-gray-800 px-4 py-3">
            <h3 className="text-sm font-medium text-gray-300">Job History</h3>
          </div>
          {(!jobs || jobs.length === 0) ? (
            <p className="px-4 py-6 text-sm text-gray-500 text-center">
              No past jobs recorded.
            </p>
          ) : (
            <div className="divide-y divide-gray-800/50">
              {[...jobs].reverse().map((job: BenchmarkStatus) => (
                <div key={job.job_id} className="flex items-center justify-between px-4 py-2.5">
                  <div className="flex items-center gap-4">
                    <span className="font-mono text-xs text-gray-400">{job.job_id}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        job.status === "completed"
                          ? "bg-green-900/50 text-green-300"
                          : job.status === "failed"
                          ? "bg-red-900/50 text-red-300"
                          : job.status === "running"
                          ? "bg-yellow-900/50 text-yellow-300"
                          : "bg-gray-800 text-gray-400"
                      }`}
                    >
                      {job.status}
                    </span>
                    {job.started_at && (
                      <span className="text-xs text-gray-500">
                        {new Date(job.started_at).toLocaleString()}
                      </span>
                    )}
                    {job.finished_at && (
                      <span className="text-xs text-gray-600">
                        → {new Date(job.finished_at).toLocaleString()}
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => handleViewLogs(job.job_id)}
                    className="rounded-lg bg-gray-800 px-3 py-1 text-xs text-gray-300 hover:bg-gray-700 transition-colors"
                  >
                    View Logs
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Viewing logs for a past job */}
        {viewingJobId && viewingLines.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400">
                Logs for job <span className="font-mono">{viewingJobId}</span>
              </span>
              <button
                onClick={() => { setViewingJobId(null); setViewingLines([]); }}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                Close
              </button>
            </div>
            <LogStream lines={viewingLines} status={viewingStatus === "loading" ? "loading" : undefined} />
          </div>
        )}
      </div>
    </div>
  );
}
