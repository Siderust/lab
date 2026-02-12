/**
 * API client — fetch helpers and WebSocket wrapper.
 */

import type {
  BenchmarkRequest,
  BenchmarkStatus,
  CompareResult,
  ExperimentResult,
  RunDetail,
  RunSummary,
} from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ----- Runs ----- //

export function fetchRuns(): Promise<RunSummary[]> {
  return get<RunSummary[]>("/runs");
}

export function fetchRun(runId: string): Promise<RunDetail> {
  return get<RunDetail>(`/runs/${runId}`);
}

export function fetchExperiment(
  runId: string,
  experiment: string
): Promise<ExperimentResult[]> {
  return get<ExperimentResult[]>(`/runs/${runId}/experiments/${experiment}`);
}

export function reloadRuns(): Promise<{ status: string }> {
  return post<{ status: string }>("/runs/reload", {});
}

// ----- Compare ----- //

export function fetchCompare(
  runA: string,
  runB: string
): Promise<CompareResult> {
  return get<CompareResult>(`/compare?run_a=${runA}&run_b=${runB}`);
}

// ----- Benchmark ----- //

export function startBenchmark(
  request: BenchmarkRequest
): Promise<BenchmarkStatus> {
  return post<BenchmarkStatus>("/benchmark/start", request);
}

export function fetchJobs(): Promise<BenchmarkStatus[]> {
  return get<BenchmarkStatus[]>("/benchmark/jobs");
}

/**
 * Connect to the benchmark WebSocket and call `onLine` for each log line.
 * Returns a close function.
 */
export function subscribeBenchmarkLogs(
  jobId: string,
  onLine: (line: string) => void,
  onDone: (status: string) => void
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/api/benchmark/ws/${jobId}`);

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "log") {
        onLine(msg.line);
      } else if (msg.type === "done") {
        onDone(msg.status);
      }
    } catch {
      onLine(ev.data);
    }
  };

  ws.onerror = () => onDone("error");
  ws.onclose = () => onDone("closed");

  return () => ws.close();
}
