# Lab Report — frame_rotation_bpn

Generated: 2026-02-12 18:09 UTC
Reference: **erfa**

## Summary

**astropy** matches erfa within p99 = 4.35 mas; median latency = 47,536 ns; speedup = 1.0×; **siderust** matches erfa within p99 = 83.50 mas; median latency = 1,219 ns; speedup = 37.9×

## Top-line Metrics

| Library | Angular error (mas) p50 | p90 | p99 | max | NaN | Inf | Latency (ns) | Throughput | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| astropy | 0.000 | 0.000 | 4.347 | 6.147 | 0 | 0 | 47536.0 | 21,037 | 1.0× |
| siderust | 44.007 | 72.082 | 83.497 | 230.539 | 0 | 0 | 1219.3 | 820,127 | 37.9× |

## Worst-N Outliers

| Library | Case | JD(TT) | Error (mas) |
|---|---|---|---|
| astropy | case_0 | 2452827.567385 | 6.147 |
| astropy | case_1 | 2457687.086734 | 6.147 |
| astropy | case_2 | 2453140.222349 | 5.324 |
| astropy | case_3 | 2456355.727242 | 5.324 |
| astropy | case_4 | 2454861.433355 | 5.324 |
| siderust | case_0 | 2415020.000000 | 230.539 |
| siderust | case_1 | 2488069.500000 | 166.542 |
| siderust | case_2 | 2462389.182977 | 92.105 |
| siderust | case_3 | 2462383.424877 | 91.951 |
| siderust | case_4 | 2462395.375832 | 91.591 |

## Alignment Checklist

```json
{
  "units": {
    "angles": "radians (internal), mas for error reporting",
    "distances": "meters",
    "float_type": "f64"
  },
  "time_input": "JD (Julian Date), TT scale for precession/nutation, UT1 for sidereal time",
  "time_scales": "TT for BPN matrix; UT1\u2248TT-69.184s simplified",
  "leap_seconds": "not applicable (JD input, no UTC conversion in this experiment)",
  "earth_orientation": {
    "ut1_minus_utc": "not used (JD(TT) input)",
    "polar_motion_xp_yp": "zero (not applied)",
    "eop_mode": "disabled"
  },
  "geodesy": "not applicable (direction-only experiment)",
  "refraction": "disabled",
  "ephemeris_source": "not applicable (no aberration/parallax)",
  "models": {
    "erfa": "IAU 2006/2000A bias-precession-nutation (eraPnm06a)",
    "siderust": "IERS 2003 frame bias + Meeus precession (\u03b6,z,\u03b8) + IAU 1980 nutation (63 terms)",
    "astropy": "IAU 2006/2000A via bundled ERFA (erfa.pnm06a)"
  },
  "mode": "common_denominator",
  "note": "ERFA and Astropy use the same IAU 2006/2000A model (reference). Siderust uses Meeus precession + IAU 1980 nutation (lower fidelity). Differences measure the model gap, not implementation bugs."
}
```

## Run Metadata

```json
{
  "date": "2026-02-12T17:55:18Z",
  "git_shas": {
    "lab": "7328c9f3e9ed",
    "siderust": "7e2e9213ef41",
    "erfa": "1d9738bed995"
  },
  "cpu": "x86_64",
  "os": "Linux 6.17.0-14-generic",
  "toolchain": {
    "python": "3.12.3",
    "rustc": "rustc 1.91.1 (ed61e7d7e 2025-11-07)",
    "cc": "gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
  }
}
```

## Reproduction

```bash
# Run date: 2026-02-12T17:55:18Z
# OS: Linux 6.17.0-14-generic  CPU: x86_64
# lab SHA: 7328c9f3e9ed
# siderust SHA: 7e2e9213ef41
# erfa SHA: 1d9738bed995

cd /path/to/siderust/lab
git submodule update --init --recursive
bash run.sh  # or: python3 pipeline/orchestrator.py --experiment frame_rotation_bpn --n 10000 --seed 42
```
