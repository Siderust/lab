# Performance Benchmarking Framework

## Overview

This framework extends the accuracy-focused experiment suite with comprehensive performance benchmarking capabilities. All 7 experimental modules now support runtime measurement, throughput calculation, and comparative performance analysis across the four reference libraries (erfa, astropy, libnova, siderust).

## Architecture

### Orchestration Layer
**File**: `pipeline/orchestrator.py`

Each experiment supports two modes:
- **Accuracy mode** (default): Measures numerical precision against reference implementations
- **Performance mode** (enabled via CLI): Measures execution time and throughput

The `--no-perf` flag skips performance measurement, running only accuracy tests.

### Performance Measurement Pattern

All performance tests follow this pattern:

1. **Input Preparation**: Generate or load the same inputs used for accuracy testing
2. **Warm-up Phase**: Run 100 iterations (or `min(N, 100)`) to warm up caches and branch predictors
3. **Timed Run**: Execute `N` operations with high-resolution timing
4. **Metrics Calculation**: Compute per-operation latency and throughput

### Adapter Implementation

Each adapter (erfa, astropy, libnova, siderust) implements performance functions for all 7 experiments:

#### C Adapters (erfa, libnova)
- Use `clock_gettime(CLOCK_MONOTONIC)` for nanosecond-precision timing
- Pre-allocate input arrays to avoid measurement contamination
- Output JSON with metrics: `per_op_ns`, `throughput_ops_s`, `total_ns`, `count`

#### Rust Adapter (siderust)
- Use `std::time::Instant::now()` for high-resolution timing
- Employ `std::hint::black_box()` to prevent optimizer dead-code elimination
- Same JSON output format as C adapters

#### Python Adapter (astropy)
- Use `time.perf_counter_ns()` for nanosecond timing
- Pre-load inputs as lists to minimize measurement overhead
- Same JSON output format

## Supported Experiments

All 7 experiments support performance benchmarking:

### 1. Frame Rotation BPN (`frame_rotation_bpn_perf`)
**Measures**: Bias-precession-nutation matrix computation time  
**Metrics**: Time to compute ICRS → True-of-Date rotation matrix  
**Input**: Julian Date (JD)  
**Libraries**: erfa, astropy, libnova, siderust

### 2. GMST/ERA (`gmst_era_perf`)
**Measures**: Greenwich Mean Sidereal Time and Earth Rotation Angle computation  
**Metrics**: Time to calculate GMST and ERA from JD  
**Input**: Julian Date (JD)  
**Libraries**: erfa, astropy, libnova, siderust

### 3. Equatorial ↔ Ecliptic (`equ_ecl_perf`)
**Measures**: Coordinate frame transformation time  
**Metrics**: Time to transform from equatorial (ICRS) to ecliptic coordinates  
**Input**: JD, Right Ascension (rad), Declination (rad)  
**Libraries**: erfa, astropy, libnova, siderust

### 4. Equatorial → Horizontal (`equ_horizontal_perf`)
**Measures**: Topocentric coordinate transformation time  
**Metrics**: Time to transform equatorial to horizontal (Alt/Az) coordinates  
**Input**: JD (UT1), JD (TT), RA (rad), Dec (rad), Longitude (rad), Latitude (rad)  
**Libraries**: erfa, astropy, libnova, siderust

### 5. Solar Position (`solar_position_perf`)
**Measures**: Sun geocentric position computation time  
**Metrics**: Time to compute apparent solar RA/Dec  
**Input**: Julian Date (JD)  
**Libraries**: erfa, astropy, libnova, siderust

### 6. Lunar Position (`lunar_position_perf`)
**Measures**: Moon geocentric/topocentric position computation time  
**Metrics**: Time to compute apparent lunar RA/Dec  
**Input**: Julian Date (JD)  
**Libraries**: erfa, astropy, libnova, siderust

### 7. Kepler Solver (`kepler_solver_perf`)
**Measures**: Kepler equation solution time  
**Metrics**: Time to solve M → E (mean anomaly to eccentric anomaly)  
**Input**: Mean anomaly (rad), Eccentricity (dimensionless)  
**Libraries**: erfa, astropy, libnova, siderust

## Usage

### Command Line

Run all experiments with performance benchmarking (default):
```bash
./run.sh run
```

Run with custom sample size:
```bash
./run.sh run 5000
```

Run specific experiments:
```bash
python3 pipeline/orchestrator.py --experiments gmst_era,kepler_solver --n 2000 --seed 42
```

Skip performance measurements (accuracy only):
```bash
python3 pipeline/orchestrator.py --experiments all --n 1000 --seed 42 --no-perf
```

Run single experiment for quick performance check:
```bash
python3 pipeline/orchestrator.py --experiments frame_rotation_bpn --n 100
```

### Webapp Integration

Performance data is automatically available in the web dashboard:

1. **Start backend** (from repo root):
   ```bash
   source .venv/bin/activate
   uvicorn webapp.backend.app.main:app --reload
   ```

2. **Start frontend** (from `webapp/frontend/`):
   ```bash
   npm install
   npm run dev
   ```

3. **Navigate to** `http://localhost:5173` and select any experiment
4. **Performance Tab** displays:
   - Per-operation latency (ns/op)
   - Throughput (operations/second)
   - Comparative bar charts across all libraries
   - Speedup factors vs. reference library

## Output Format

### JSON Result Schema

Each library's result file (`results/<date>/<experiment>/<library>.json`) contains:

```json
{
  "experiment": "gmst_era",
  "candidate_library": "siderust",
  "reference_library": "erfa",
  "alignment": { ... },
  "inputs": { "count": 100, "seed": 42 },
  "accuracy": { ... },
  "performance": {
    "per_op_ns": 159.0,
    "throughput_ops_s": 6289308,
    "total_ns": 15900,
    "batch_size": 100
  },
  "run_metadata": { ... },
  "reference_performance": {
    "per_op_ns": 235.0,
    "throughput_ops_s": 4255138
  }
}
```

**Fields**:
- `performance`: Candidate library's metrics
  - `per_op_ns`: Average time per operation (nanoseconds)
  - `throughput_ops_s`: Operations per second
  - `total_ns`: Total elapsed time for all operations
  - `batch_size`: Number of operations measured
- `reference_performance`: Reference library (erfa) metrics for comparison

### Summary Reports

Markdown reports (`reports/<date>/<experiment>/summary.md`) include performance sections:

```markdown
### Performance Comparison

| Library | Per-op (ns) | Throughput (ops/s) | Speedup vs ref |
|---------|-------------|-------------------|----------------|
| erfa    | 235.0       | 4,255,138         | 1.0×           |
| siderust| 159.0       | 6,289,308         | 1.5×           |
| astropy | 1535.0      | 651,542           | 0.15×          |
| libnova | 18.0        | 57,273,769        | 24.4×          |
```

## Performance Characteristics by Library

### ERFA (Reference)
- **Language**: C
- **Optimization**: `-O3` with GCC
- **Characteristics**: Baseline performance, well-optimized IAU-standard routines

### Siderust
- **Language**: Rust
- **Optimization**: `--release` (equivalent to `-O3` + LTO)
- **Characteristics**: Zero-cost abstractions, strong type safety, competitive with C

### Astropy
- **Language**: Python (with ERFA bindings via ctypes/Cython)
- **Optimization**: Depends on ERFA C library performance
- **Characteristics**: Python overhead, but delegates heavy computation to C

### Libnova
- **Language**: C
- **Optimization**: `-O3` with GCC
- **Characteristics**: Different algorithms (Meeus vs. IAU), sometimes faster due to simplifications

## Interpreting Results

### Metrics Meaning

- **Per-operation latency (ns/op)**: Lower is better. Measures average time for a single operation.
- **Throughput (ops/s)**: Higher is better. Measures how many operations per second.
- **Speedup vs reference**: Ratio of reference library performance to candidate library. >1.0× means faster than reference.

### Typical Performance Ranges

| Experiment | Typical Range (ns/op) | Notes |
|------------|----------------------|-------|
| frame_rotation_bpn | 2,000 - 90,000 | Matrix computation heavy |
| gmst_era | 18 - 1,500 | Simple polynomial evaluation |
| equ_ecl | 50 - 400 | Trigonometric transforms |
| equ_horizontal | 100 - 800 | Spherical trigonometry |
| solar_position | 300 - 2,000 | VSOP87 series evaluation |
| lunar_position | 1,000 - 10,000 | ELP 2000 series (many terms) |
| kepler_solver | 300 - 2,500 | Iterative Newton-Raphson |

### Accuracy vs. Performance Trade-offs

Some libraries trade accuracy for speed:
- **libnova**: Often faster due to Meeus approximations, but lower accuracy (arcsec-level errors)
- **erfa/astropy**: IAU-standard precision, moderate performance
- **siderust**: Competitive performance with good accuracy (implementation-dependent)

## Advanced Topics

### Measurement Precision

- **Warm-up phase**: Ensures CPU caches, branch predictors, and JIT optimizations are stable
- **Batch measurement**: Reduces timing overhead by measuring many operations together
- **Monotonic clocks**: Prevents clock skew from system time adjustments

### Avoiding Compiler Optimizations

All adapters use techniques to prevent dead-code elimination:
- **Rust**: `std::hint::black_box(&result)` forces optimizer to assume result is used
- **C**: Volatile variables or external function calls to prevent DCE
- **Python**: Accumulator pattern (result sink) prevents optimization

### Reproducibility

Performance may vary due to:
- **CPU frequency scaling**: Disable with `cpupower frequency-set -g performance`
- **Thermal throttling**: Ensure adequate cooling during long runs
- **Background processes**: Close unnecessary applications
- **Turbo boost**: May cause variability; consider disabling for benchmarks

For production benchmarking, consider:
```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set -g performance

# Pin to specific CPU cores
taskset -c 0 python3 pipeline/orchestrator.py --experiments all --n 10000
```

## Extending the Framework

### Adding Performance to New Experiments

1. **Orchestrator** (`pipeline/orchestrator.py`):
   - Add `format_<experiment>_perf_input()` function
   - Modify `run_experiment_<experiment>()` to accept `run_perf` parameter
   - Add performance measurement section following existing patterns

2. **Adapters**:
   - **ERFA** (`pipeline/adapters/erfa_adapter/main.c`):
     - Add `run_<experiment>_perf()` function
     - Use `clock_gettime(CLOCK_MONOTONIC)` for timing
     - Update `main()` dispatcher
   
   - **Astropy** (`pipeline/adapters/astropy_adapter/adapter.py`):
     - Add `run_<experiment>_perf()` function
     - Use `time.perf_counter_ns()` for timing
     - Update dispatch dictionary
   
   - **Libnova** (`pipeline/adapters/libnova_adapter/main.c`):
     - Add `run_<experiment>_perf()` function
     - Follow ERFA adapter pattern
   
   - **Siderust** (`pipeline/adapters/siderust_adapter/src/main.rs`):
     - Add `run_<experiment>_perf()` function
     - Use `Instant::now()` for timing
     - Update `main()` match statement

3. **Schema validation**:
   - Ensure JSON output matches `PerformanceData` model in `webapp/backend/app/models/schemas.py`

4. **Frontend**:
   - Performance visualization is automatic via `PerformanceTab` component
   - No changes needed unless adding custom visualizations

## Validation Checklist

After implementing performance for a new experiment:

- [ ] Run experiment: `python3 pipeline/orchestrator.py --experiments <name> --n 100`
- [ ] Verify JSON files in `results/<date>/<experiment>/` have `performance` and `reference_performance` fields
- [ ] Check summary report in `reports/<date>/<experiment>/summary.md` includes performance table
- [ ] Confirm webapp backend loads results: `curl http://localhost:8000/api/runs`
- [ ] Verify frontend Performance tab displays charts
- [ ] Run with larger N to verify timing stability: `--n 10000`

## Known Limitations

1. **Python overhead**: Astropy adapter includes Python interpreter overhead not present in pure C/Rust implementations
2. **Small N instability**: For `N < 100`, timing resolution may introduce variance; use `N ≥ 1000` for stable benchmarks
3. **System variance**: First run after boot may show different performance than subsequent runs (cache warming)
4. **Library versions**: Performance characteristics depend on compiler versions, optimization flags, and dependency versions

## References

- Clock precision: `man clock_gettime` (Linux), `std::time::Instant` (Rust)
- Compiler optimizations: GCC `-O3` flags, Rust release profile settings
- Benchmarking best practices: [Rust Performance Book](https://nnethercote.github.io/perf-book/), [Google Benchmark](https://github.com/google/benchmark)
