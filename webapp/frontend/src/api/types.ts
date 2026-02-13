/** TypeScript interfaces matching the backend Pydantic schemas. */

export interface PercentileStats {
  p50: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
  min: number | null;
  mean: number | null;
  rms: number | null;
}

export interface WorstCase {
  jd_tt: number | null;
  angular_error_mas: number | null;
}

export interface PerformanceData {
  per_op_ns: number | null;
  per_op_ns_mean: number | null;
  per_op_ns_median: number | null;
  per_op_ns_std_dev: number | null;
  per_op_ns_min: number | null;
  per_op_ns_max: number | null;
  per_op_ns_ci95: [number, number] | null;
  per_op_ns_cv_pct: number | null;
  throughput_ops_s: number | null;
  total_ns: number | null;
  total_ns_median: number | null;
  batch_size: number | null;
  rounds: number | null;
  samples: number[] | null;
  warnings: string[] | null;
}

export interface ExperimentDescription {
  title?: string;
  what?: string;
  why?: string;
  units?: string;
  interpret?: string;
}

export interface BenchmarkConfig {
  perf_rounds?: number;
  perf_warmup?: number;
  perf_enabled?: boolean;
}

export interface AlignmentChecklist {
  units?: Record<string, string>;
  time_input?: string;
  time_scales?: string;
  leap_seconds?: string;
  earth_orientation?: Record<string, string>;
  geodesy?: string;
  refraction?: string;
  ephemeris_source?: string;
  models?: Record<string, string>;
  mode?: string;
  note?: string;
  [key: string]: unknown;
}

export interface RunMetadata {
  date: string | null;
  git_shas: Record<string, string>;
  git_branch: string | null;
  cpu: string | null;
  cpu_model: string | null;
  cpu_count: number | null;
  os: string | null;
  platform_detail: string | null;
  toolchain: Record<string, string>;
}

export interface ExperimentResult {
  experiment: string;
  candidate_library: string;
  reference_library: string;
  description: ExperimentDescription | Record<string, unknown>;
  alignment: AlignmentChecklist | null;
  inputs: Record<string, unknown>;
  accuracy: Record<string, unknown>;
  performance: PerformanceData | Record<string, unknown>;
  reference_performance: PerformanceData | Record<string, unknown>;
  benchmark_config: BenchmarkConfig | Record<string, unknown>;
  run_metadata: RunMetadata | null;
}

export interface RunSummary {
  id: string;
  timestamp: string | null;
  git_shas: Record<string, string>;
  machine: string | null;
  experiments: string[];
  libraries: string[];
  result_count: number;
  has_reports: boolean;
}

export interface RunDetail {
  id: string;
  timestamp: string | null;
  git_shas: Record<string, string>;
  machine: string | null;
  experiments: Record<string, ExperimentResult[]>;
  has_reports: Record<string, boolean>;
}

export interface MetricDelta {
  experiment: string;
  library: string;
  metric: string;
  value_a: number | null;
  value_b: number | null;
  delta: number | null;
  delta_pct: number | null;
  regression: boolean;
}

export interface CompareResult {
  run_a: string;
  run_b: string;
  deltas: MetricDelta[];
}

export interface BenchmarkRequest {
  experiments: string[];
  n: number;
  seed: number;
  no_perf: boolean;
  perf_rounds: number;
  ci_mode: boolean;
  notes: string;
}

export interface BenchmarkStatus {
  job_id: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  run_id: string | null;
  config: BenchmarkRequest | null;
  current_experiment: string | null;
  current_step: string | null;
  progress_pct: number | null;
}

/** Consistent color mapping for libraries. */
export const LIBRARY_COLORS: Record<string, string> = {
  erfa: "#3b82f6",       // blue
  siderust: "#f97316",   // orange
  astropy: "#22c55e",    // green
  libnova: "#ef4444",    // red
};

export const ALL_EXPERIMENTS = [
  "frame_rotation_bpn",
  "gmst_era",
  "equ_ecl",
  "equ_horizontal",
  "solar_position",
  "lunar_position",
  "kepler_solver",
] as const;
