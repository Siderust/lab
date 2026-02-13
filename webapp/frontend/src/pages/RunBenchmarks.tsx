import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { startBenchmark, subscribeBenchmarkLogs, fetchJobs, fetchJobLogs, cancelJob } from "../api/client";
import type { BenchmarkRequest, BenchmarkStatus } from "../api/types";
import Header from "../components/layout/Header";
import BenchmarkForm from "../components/benchmark/BenchmarkForm";
import LogStream from "../components/benchmark/LogStream";
import { Loader2, X, Settings2 } from "lucide-react";

export default function RunBenchmarks() {
  const navigate = useNavigate();
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [isCancelling, setIsCancelling] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [currentConfig, setCurrentConfig] = useState<BenchmarkRequest | null>(null);

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
      setLogLines([]);
      setCurrentConfig(request);

      try {
        const job = await startBenchmark(request);
        setJobId(job.job_id);
        setStatus("running");

        subscribeBenchmarkLogs(
          job.job_id,
          (line) => {
            setLogLines((prev) => [...prev, line]);
          },
          (finalStatus) => {
            setStatus(finalStatus);
            if (finalStatus === "completed") {
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

  const handleCancel = useCallback(async () => {
    if (!jobId || isCancelling) return;
    setIsCancelling(true);
    try {
      await cancelJob(jobId);
      setStatus("cancelled");
    } catch (err) {
      console.error("Failed to cancel job:", err);
      setStatus("error");
    } finally {
      setIsCancelling(false);
    }
  }, [jobId, isCancelling]);

  const isRunning = status === "running" || status === "starting";

  return (
    <div>
      <Header
        title="Run Benchmarks"
        subtitle="Select experiments, set parameters, and launch."
      />

      <div className="space-y-6">
        <div className="space-y-4">
          <BenchmarkForm onSubmit={handleSubmit} disabled={isRunning} />
          
          {/* Loading indicator and cancel button */}
          {isRunning && (
            <div className="space-y-3">
              <div className="flex items-center justify-between rounded-xl border border-gray-800 bg-gray-900/60 px-6 py-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin text-orange-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-300">
                      Benchmark running...
                    </p>
                    {jobId && (
                      <p className="text-xs text-gray-500 font-mono mt-0.5">
                        Job ID: {jobId}
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={handleCancel}
                  disabled={isCancelling}
                  className="flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <X className="h-4 w-4" />
                  {isCancelling ? "Cancelling..." : "Cancel Execution"}
                </button>
              </div>

              {/* Active config display */}
              {currentConfig && (
                <div className="rounded-xl border border-gray-800 bg-gray-900/40 px-4 py-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Settings2 className="h-4 w-4 text-gray-400" />
                    <span className="text-xs font-medium uppercase text-gray-400">Run Configuration</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div>
                      <span className="text-gray-500">Experiments:</span>{" "}
                      <span className="text-gray-300">{currentConfig.experiments.join(", ")}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">N:</span>{" "}
                      <span className="text-gray-300 font-mono">{currentConfig.n}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Seed:</span>{" "}
                      <span className="text-gray-300 font-mono">{currentConfig.seed}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Perf:</span>{" "}
                      <span className="text-gray-300">
                        {currentConfig.no_perf ? "disabled" : `${currentConfig.perf_rounds} rounds`}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Live log stream */}
              {logLines.length > 0 && (
                <LogStream lines={logLines} status="running" />
              )}
            </div>
          )}
        </div>

        {status === "completed" && (
          <p className="text-green-400 text-sm">
            Benchmark completed successfully. Redirecting to runs list...
          </p>
        )}
        {status === "failed" && (
          <div className="space-y-2">
            <p className="text-red-400 text-sm">
              Benchmark failed. Check the logs below for details.
            </p>
            {logLines.length > 0 && <LogStream lines={logLines} status="failed" />}
          </div>
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
        {status === "cancelled" && (
          <p className="text-yellow-400 text-sm">
            Benchmark execution cancelled.
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
                          : job.status === "cancelled"
                          ? "bg-orange-900/50 text-orange-300"
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
