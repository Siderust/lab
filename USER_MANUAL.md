# Siderust Lab — Reports & GUI Design (User Manual)

This document defines:

1. **What report artifacts exist** (what gets written to `results/` and `reports/`)
2. **How to compare libraries fairly** (accuracy/correctness and performance tracks)
3. **How to illustrate results** (plots, tables, matrices)
4. A **schematic GUI design** for a web app that runs benchmarks and visualizes comparisons

---

## 0) Goals

The lab exists to answer, with reproducible evidence:

- *Correctness/accuracy*: “How close is library X to a trusted reference for a defined model?”
- *Performance*: “What is the cost (latency/throughput/memory) per primitive and per end-to-end workload?”

The web app should make it easy to:

- Run benchmarks with pinned assumptions
- Compare tools across experiments (and across runs)
- Explain *why* results differ (model parity + outliers + drift plots)

---

## 1) Terminology

- **Run**: one execution of the pipeline on a machine at a point in time (includes environment metadata).
- **Experiment**: a benchmark family (e.g., `gmst_era`, `frame_rotation_bpn`).
- **Case / test vector**: one input item inside an experiment (an epoch, a direction, an observer site…).
- **Reference**: the tool/model chosen as “truth” for scoring (often ERFA/SOFA-derived).
- **Candidate**: the tool being compared to the reference.
- **Mode**: named set of assumptions for parity (e.g., `common_denominator` vs `high_fidelity`).

---

## 2) What reports we have (artifacts on disk)

### A) Machine-readable results (`results/`)

**Folder layout**

```
results/<YYYY-MM-DD>/<experiment>/
  <library>.json
  summary.md
```

**What’s inside `<library>.json`**

- `experiment`: experiment id (string)
- `candidate_library`, `reference_library`
- `alignment`: the **alignment checklist** (units, time scales, models, assumptions)
- `inputs`: counts/seeds and high-level input range
- `accuracy`: experiment-specific accuracy metrics + outlier list
- `performance`: microbenchmark data when available (e.g., `per_op_ns`, `throughput_ops_s`)
- `reference_performance`: reference timing (when measured)
- `run_metadata`: date, git SHAs, OS/CPU/toolchain

**What `summary.md` is**

A compact Markdown table (or several) with “top-line” numbers per candidate library:

- p50/p99/max error (domain-appropriate)
- optional performance columns (ns/op, speedup vs reference)
- plus a “Feature / Model parity matrix” section for context

These results are the primary data source for the GUI.

### B) Human-friendly “lab reports” (`reports/`)

**Folder layout**

```
reports/<YYYY-MM-DD>/<experiment>/
  report.md
  index.html
```

**What’s inside**

- A narrative summary (“p99 error within …”, “speedup …×”)
- Top-line metrics tables
- Worst-N outlier tables
- Embedded plots (often rendered into the HTML)
- Alignment checklist + run metadata + reproduction commands

These are good for sharing as static artifacts. The GUI should be able to deep-link to them, but should not *depend* on them.

### C) Recommended “next artifacts” (to enable richer GUI views)

To support drill-down and plot reconstruction (without re-running), add these per run:

- `vectors/` (inputs actually used, with case ids)
- `per_case.parquet` or `per_case.csv` (per-case errors + any residuals/invariants)
- `env.json` (run metadata once per run, rather than duplicated per file)
- `bench_results.csv` (long-form metrics: tool, experiment, N, metric, value)

---

## 3) How we should compare reports (fairness rules)

Comparisons are only meaningful if the experiment locks down assumptions.

### A) Always include an alignment checklist

Each experiment must explicitly state:

- **Units & conventions**: degrees/radians, meters/km/AU, axis order, handedness, azimuth convention
- **Time**: JD/MJD/seconds; required scales (UTC/TAI/TT/UT1/TDB); leap second source
- **Earth orientation**: UT1−UTC, polar motion, EOP enabled vs forced to zero
- **Geodesy**: ellipsoid (WGS84), height model
- **Astronomical models**: precession/nutation model identifiers; sidereal time convention
- **Physics toggles**: aberration/light-time, refraction
- **Ephemeris**: analytic vs JPL kernel (DE version), interpolation settings

If parity is impossible, split the report into named **modes**:

- `common_denominator`: everyone runs the simplest shared model
- `high_fidelity`: best-available model per tool, but comparisons are “tradeoffs” not “who is right”

### B) Two benchmark tracks (always)

1) **Accuracy/Correctness** (vs trusted reference *or* invariants/residuals)
2) **Performance** (micro: ns/op; macro: end-to-end pipelines)

Avoid “accuracy fights” when tools are doing different physics: document the model gap and score separately.

### C) Statistics to report

**Accuracy**

- `median`, `RMS`, `p95`, `p99`, `max`
- signed bias (mean signed error) + absolute error distributions
- NaN/Inf / non-convergence counts
- for orbit kernels: invariant drift rates (energy, angular momentum)

**Performance**

- median + `p95` latency
- throughput (points/s) for batched workloads
- allocations/op and peak RSS when measurable
- scaling curves vs batch size

Record machine + version context (CPU, OS, compiler/interpreter, git SHAs).

---

## 4) How we should illustrate results (plots, tables, matrices)

The goal is “read it in 10 seconds” clarity. Prefer a small, consistent plot set across experiments.

### A) Must-have tables/matrices

1) **Feature + model parity matrix**

For each experiment (and ideally one global matrix per run):

- Frames supported / definitions
- Ephemeris/model used
- Aberration/light-time/refraction toggles
- Earth orientation inputs (UT1−UTC, polar motion)
- Vectorization/batching/threading support

2) **Score tables per experiment family**

One table each for: Time, Frames, Ephemerides, Orbits.

Columns (suggested):

- accuracy: p50/p99/max + fail counts
- performance: median/p95 + throughput + memory

3) **Regression guard table**

Pick ~20 canonical vectors per experiment and store expected outputs for CI gating (later).

### B) Accuracy plots (best set)

1) **CDF of absolute error** (one curve per tool)

- Angular experiments: CDF of `|sep|` (mas or arcsec)
- Time experiments: CDF of `|Δt|` (ns/µs)

2) **Error vs epoch**

Line or scatter plot:

- highlights leap-second issues, long-term drift, model transitions

3) **Sky/parameter heatmaps** (when inputs are spherical grids)

- RA/Dec heatmap colored by error
- reveals quadrant flips, pole issues, convention mismatches

4) **Orbit invariants drift**

For propagators:

- energy and angular momentum vs time
- annotate drift rate per orbit/day

### C) Performance plots (best set)

1) **Runtime vs batch size (log–log)**

Shows overhead vs throughput and where vectorization helps.

2) **Bar chart of ns/op** for key primitives

Median with p95 whiskers.

3) **Memory vs batch size**

Especially important for Rust vs Python comparisons.

### D) “At-a-glance” Pareto plots

For each experiment family:

- x-axis: p95/p99 absolute error (log scale)
- y-axis: p95 latency or ns/op (log scale)
- one point per tool (optionally per “mode”)

---

## 5) Schematic design of the GUI (web app)

### A) Information architecture (pages)

1) **Runs**

- List of past runs (timestamp, git SHAs, machine, tags)
- Actions: “view”, “compare”, “download artifacts”

2) **Run Overview (Dashboard)**

- Global summary table (experiments × libraries)
- Global feature/model parity matrix
- Pareto scatter per experiment family
- Alerts: missing parity, missing perf, high NaN rates

3) **Experiment Detail**

Tabs:

- **Overview**: top-line metrics cards + parity statement
- **Accuracy**: CDF + error-vs-epoch + bias summaries + heatmaps (when applicable)
- **Performance**: ns/op + throughput + scaling curves + memory (if available)
- **Outliers**: worst-N table; click-through case explorer
- **Assumptions**: alignment checklist (diffable across runs)

4) **Compare Runs**

- Pick two runs (A/B) and view:
  - metric deltas per experiment and per library
  - regression highlights (“p99 error worsened by …”, “latency +…%”)

5) **Run Benchmarks**

- Select experiments, libraries, `N`, seed, and mode
- Toggle performance tests (micro + macro)
- Show live progress + logs
- Persist run notes (why this run exists)

### B) Primary UX flows

1) **Run → Review → Share**

- run benchmarks
- view dashboard and experiment drill-down
- export `results/` + static `reports/` bundle

2) **Debug a disagreement**

- start from Pareto/scatter outlier
- drill into experiment → outliers
- compare raw inputs/outputs (requires storing vectors/per-case data)
- inspect parity checklist (often the real reason)

3) **Regression check**

- compare two runs across commits
- filter to “changed metrics only”

### C) Wireframe (schematic)

**Run overview**

```
┌──────────────────────────────────────────────────────────────────┐
│ Siderust Lab  ┆ Runs ▸ 2026-02-12 21:32Z                          │
├──────────────────────────────────────────────────────────────────┤
│ [Run] [Compare] [Download]  Mode: common_denominator              │
├──────────────────────────────────────────────────────────────────┤
│ Summary Table (experiments × libraries)                           │
│  - p99 error, max error, ns/op, speedup                           │
├──────────────────────────────────────────────────────────────────┤
│ Pareto (Frames)     Pareto (Time)      Pareto (Ephemerides)       │
│ (error vs latency)  (error vs latency) (error vs latency)         │
├──────────────────────────────────────────────────────────────────┤
│ Feature/Model Parity Matrix                                       │
│ (frames, ephemeris, EOP, refraction, aberration)                  │
└──────────────────────────────────────────────────────────────────┘
```

**Experiment detail**

```
┌──────────────────────────────────────────────────────────────────┐
│ Experiment: frame_rotation_bpn   Reference: erfa   Mode: common…  │
├──────────────────────────────────────────────────────────────────┤
│ Tabs: [Overview] [Accuracy] [Performance] [Outliers] [Assumptions]│
├──────────────────────────────────────────────────────────────────┤
│ Overview:                                                         │
│  - Metric cards (p50/p99/max, NaN/Inf)                             │
│  - “Parity” callout (what models differ)                           │
├──────────────────────────────────────────────────────────────────┤
│ Accuracy:                                                         │
│  - CDF(|error|)     - Error vs epoch                               │
│  - Bias histogram   - Heatmap (if spherical grid)                  │
├──────────────────────────────────────────────────────────────────┤
│ Outliers:                                                         │
│  - Worst-N table (case id, epoch, error) → click → Case Explorer   │
└──────────────────────────────────────────────────────────────────┘
```

### D) Data model (minimal)

The GUI should treat a **run** as the unit of organization.

- `Run`: id, timestamp, git SHAs, machine metadata, notes/tags
- `ExperimentResult`: run_id, experiment id, mode, reference tool, candidate tool
- `Metric`: name, units, summary stats, per-case series (optional but ideal)
- `Parity`: a structured object (alignment checklist), diffable across runs

### E) Execution architecture (recommended)

To support “run benchmarks from the UI” safely:

- **Backend**: a local server that can:
  - spawn `python3 pipeline/orchestrator.py …` (and later macro workloads)
  - stream logs + progress (polling or WebSocket)
  - register outputs as a new `Run`
- **Frontend**: reads run/experiment JSON via a simple API and renders plots/tables.

If the app must be “static only”, it can still **browse** existing `results/` folders, but cannot reliably execute benchmarks without a backend.

---

## 6) What “good” looks like (minimum viable GUI)

MVP checklist:

- Browse runs; open a run dashboard
- Per-experiment detail view with:
  - top-line metrics table
  - CDF + error-vs-epoch (when series available)
  - performance plots when available
  - parity checklist display
- Compare two runs (delta table + regression flags)
- Export/download a run’s artifacts

---

## 7) Appendix: current experiments in `pipeline/orchestrator.py`

Implemented today:

- `frame_rotation_bpn` — BPN direction transform + optional perf timing
- `gmst_era` — sidereal time + Earth rotation angle comparison
- `equ_ecl` — equatorial ↔ ecliptic transform
- `equ_horizontal` — equatorial → horizontal (Alt/Az)
- `solar_position` — Sun apparent/geocentric position (model-dependent)
- `lunar_position` — Moon position (model-dependent)
- `kepler_solver` — Kepler solver residuals/self-consistency

Tools currently wired:

- `erfa` (reference)
- `siderust`
- `astropy`
- `libnova`
- `anise` (available for supported experiments; unsupported ones are marked skipped)

Planned extensions:

- add orbit propagation + Lambert experiments
- add macrobenchmarks (end-to-end pipelines) and memory/allocation reporting


# Backend
cd webapp/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (dev mode)
cd webapp/frontend
npm install
npm run dev     # Vite dev server on :5173, proxies /api to :8000

# Production (single server)
cd webapp/frontend && npm run build   # produces dist/
cd ../backend && uvicorn app.main:app --port 8000   # serves API + frontend
