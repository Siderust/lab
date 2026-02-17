# Astro-Tools Benchmark Laboratory

This repository is a benchmarking and validation “laboratory” for comparing **Siderust** against a small set of established astronomy/astrodynamics libraries that are vendored as git submodules (and implemented in different languages). The goal is to produce **empirical, reproducible** answers to questions like:

- How accurate are Siderust coordinate/time transformations compared to ERFA/Astropy?
- Where does Siderust win/lose on performance (latency, throughput, allocations)?
- Which improvements would move Siderust along the speed/accuracy Pareto frontier?

## What’s in here (today)

Git submodules (pinned by commit SHA):

- `siderust/` (Rust)
- `anise/` (Rust)
- `astropy/` (Python)
- `erfa/` (C)
- `libnova/` (C)

Initialize everything (including nested submodules inside `siderust/`):

```bash
git submodule update --init --recursive
```

## What we want to measure

This lab is intended to generate **comparable** results across libraries, which means: shared inputs, shared conventions, and a clear definition of “truth” (reference).

### Alignment checklist (to avoid misleading comparisons)

Before comparing outputs, each experiment should explicitly lock down:

- **Units & types**: radians vs degrees, meters vs kilometers, float64/float32 policy, axis ordering (x/y/z), handedness
- **Time representation**: JD/MJD/seconds, required time scales (UTC/TAI/TT/UT1), leap second source
- **Earth orientation**: UT1−UTC, polar motion (xp/yp), and whether EOP is enabled or forced to zero
- **Geodesy**: reference ellipsoid (e.g., WGS84), height definition (ellipsoidal vs geoid)
- **Astronomical models**: precession/nutation model identifiers (e.g., IAU 2000/2006), sidereal time conventions
- **Refraction policy**: disabled vs enabled, and which model/constants
- **Ephemeris source** (if applicable): analytic vs JPL kernels, kernel version, and interpolation settings

Where libraries cannot be configured to the same model, run the experiment in named “modes” (e.g., *common-denominator* vs *high-fidelity*) and document the differences.

### Accuracy / correctness

Core statistics to report per experiment (and per library):

- **Error norms** (domain-appropriate):
  - angular separation (e.g., arcsec / mas) for sky positions and frame transforms
  - position error (meters) for ECEF/ECI and topocentric vectors
  - time error (µs / ns) for time-scale conversions and event boundary times
- **Aggregate stats**: min/max, mean, median, RMS, std-dev, p50/p90/p95/p99
- **Outliers**: worst-N cases + input conditions that trigger them
- **Invariants / sanity checks**: unit-vector norms, round-trip closure (A→B→A), monotonicity, NaN/Inf counts

Where available, the reference should be **ERFA/SOFA-derived** results (often via `erfa` or `astropy`) or published test vectors (IAU/IERS/SOFA).

### Performance

Core statistics to report per experiment (and per library):

- **Latency**: per-call time (ns/µs), including cold vs warm runs where relevant
- **Throughput**: ops/s for batched workloads (N=1, 10, 1k, 1M)
- **Resource usage**: allocations/op, peak RSS, CPU utilization
- **Scaling curves**: time vs batch size, time vs epoch range, time vs model options

Reporting should emphasize robust summaries (median + dispersion) and include enough run metadata to reproduce results (CPU model, OS, compiler/interpreter versions, git SHAs).

## Experiments to implement

The concrete experiments below are the “backbone” of a proper comparative analysis. Each experiment should define:

1. **Inputs** (generator + edge-case suite)
2. **Reference** (what is considered correct, and why)
3. **Metrics** (error + performance)
4. **Adapters** (how each library is invoked consistently)

### 1) Coordinate frame transformations

Compare frame conversion outputs across libraries, for representative epochs and sky/space positions.

Examples:

- ICRS/GCRS/CIRS ↔ ITRS (as supported by each library)
- ECI ↔ ECEF (including Earth rotation, precession/nutation models where available)
- Round-trip closure tests (A→B→A) to expose numerical drift

Metrics:

- angular error between directions
- cartesian position error (m) when applicable
- closure error for round-trips (angular + cartesian)

### 2) Time scales and Earth rotation quantities

Validate time conversions and derived quantities used by other pipelines.

Examples:

- UTC ↔ TAI ↔ TT ↔ UT1 (where supported and with documented assumptions)
- GMST/GAST / Earth Rotation Angle comparisons

Metrics:

- absolute time conversion error vs reference
- drift across long epoch ranges
- leap second boundary behavior (correctness + robustness)

### 3) Topocentric Alt/Az/Elevation (and refraction policy)

For a fixed observer (lat/lon/height) and target (RA/Dec, or ECI state), compute topocentric azimuth/elevation.

Metrics:

- angular error in az/el
- elevation threshold classification accuracy (above/below)
- sensitivity to refraction model choices (if any)

### 4) “Elevation periods” / event finding

Given an elevation threshold (e.g., 10°) and a target, find the time intervals where elevation ≥ threshold.

Key comparison axes:

- stepping vs root-finding approaches
- accuracy of boundary times (rise/set crossings)
- performance as the search horizon grows

Metrics:

- boundary time error (start/end) vs reference (µs/ms)
- missed/extra intervals (precision/recall on detected windows)
- runtime vs horizon and vs sampling resolution

### 5) Robustness and edge-case suites

Not a single experiment, but a standardized set of cases that every adapter must pass:

- poles and near-poles observers
- near-horizon targets
- very small/large angles
- wide epoch spans (e.g., 1900–2100) as supported
- leap second days

Metrics:

- pass/fail + categorized failure reasons
- NaN/Inf counts and domain errors

## Making different languages comparable (adapters & containers)

Because the dependencies span Rust/Python/C, this repo is expected to grow:

- **Adapters**: small wrappers that expose a consistent API per library (inputs/outputs normalized, units explicit)
- **Orchestrators**: scripts that generate inputs, run the adapters, collect timings, and write structured results
- **Docker images** (or similar): pinned environments so the same benchmark can run on any machine/CI runner

Suggested output format:

- `results/<date>/<experiment>/<library>.json` (metrics + metadata)
- a generated summary table (Markdown/CSV) and plots (PNG/SVG)

## Desired end state

At maturity, this lab should produce:

- a repeatable “one command” benchmark run (local + CI)
- accuracy scorecards per experiment (with traceable references)
- performance dashboards (tables + plots)
- a short list of concrete, data-backed improvement targets for Siderust

## Benchmark reliability & reproducibility

The benchmark harness is designed for **reliability first**:

### Multi-sample performance measurement

Performance benchmarks run **multiple rounds** (default: 5) and report robust statistics:

- **Median** (primary metric, robust to outliers)
- **Standard deviation**, **min/max**, **coefficient of variation (CV%)**
- **95% confidence interval** assuming normal distribution
- **Individual samples** preserved for analysis

If CV% exceeds 20%, a warning is emitted. If per-op time is below the measurable threshold (10 ns), measurements are flagged as unreliable.

### Deterministic accuracy

Accuracy benchmarks are **fully deterministic** given the same `(N, seed)`:
- Input generation uses `numpy.random.default_rng(seed)` — no hidden randomness
- Each dataset is fingerprinted (SHA-256 hash) for identity verification
- Re-running with the same manifest produces identical results

### Run manifests & provenance

Every run produces a `manifest.json` capturing:
- Exact configuration (experiments, N, seed, perf_rounds)
- Full environment metadata (CPU model, core count, OS, compiler versions)
- Git SHAs for all submodules (siderust, anise, erfa, libnova)
- Git branch and numpy/pyerfa versions

### CI mode

For continuous integration, use `--ci` to run with reduced parameters:
```bash
python3 pipeline/orchestrator.py --ci --experiments all
```
This defaults to N=100 and 2 perf rounds — fast but still meaningful.

### Running sanity tests

```bash
python3 -m pytest pipeline/tests/ -v
```

Tests verify:
- Adapters produce non-zero output (no dead-code elimination)
- Input generation is deterministic
- Dataset fingerprints are reproducible
- Performance measurements are above noise threshold
- Run metadata is complete

## How to interpret results

### For non-experts

Each experiment includes a **description** explaining:
- **What** it measures (in plain English)
- **Why** it matters for astronomy
- **Units** used and their meaning
- **How to interpret** differences between libraries

These descriptions appear in the web app's experiment detail view.

### Accuracy grades

The Siderust scoreboard grades accuracy as:
- **Excellent**: errors below the typical measurement precision
- **Good**: errors within acceptable limits for most applications
- **Fair**: noticeable errors that may affect precision work
- **Poor**: significant errors that need investigation

### Performance interpretation

Performance is reported as **median ns/op** (nanoseconds per operation):
- Includes the full computation, not just the core math
- Python adapters include interpreter overhead (expected to be slower)
- Multi-sample statistics detect unstable measurements
- Speedup is relative to ERFA (the C reference implementation)

## Troubleshooting suspiciously fast benchmarks

If benchmarks report unrealistically fast times:

1. **Check the warnings**: The harness warns if per-op time < 10 ns
2. **Check CV%**: High CV (>20%) indicates measurement instability
3. **Increase N**: Larger batch sizes improve measurement accuracy
4. **Increase rounds**: More rounds reduce noise in the median
5. **Check adapters**: Verify the adapter binary was rebuilt (`--no-build` skips rebuild)
6. **Verify output sink**: All adapters use `_sink` values that are printed to stdout, preventing dead-code elimination by the compiler

Common causes of suspiciously fast benchmarks:
- Stale adapter binary (not rebuilt after code changes)
- Very small N with fast operations (measurement dominated by process startup)
- System load affecting timing (high CV% is the indicator)

## User manual

See `USER_MANUAL.md` for instructions on running the analysis pipeline and interpreting/visualizing the results.

See `PERFORMANCE_BENCHMARKING.md` for comprehensive documentation on the performance benchmarking framework, including metrics, usage, output format, and guidelines for extending the framework to new experiments.

See `EXECUTION_MANAGEMENT.md` for details on the execution management system, including unique directory naming, execution locking, cancellation mechanism, and UI state flow.
