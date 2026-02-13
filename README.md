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

## User manual

See `USER_MANUAL.md` for instructions on running the analysis pipeline and interpreting/visualizing the results.
