# Lab Report — gmst_era

Generated: 2026-02-12 18:09 UTC
Reference: **erfa**

## Summary

**astropy** matches erfa within p99 = 0.00 arcsec; **siderust** matches erfa within p99 = 0.07 arcsec

## Top-line Metrics

| Library | GMST error (arcsec) p50 | p90 | p99 | max | NaN | Inf | Latency (ns) | Throughput | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| astropy | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | — | — | — |
| siderust | 0.027 | 0.060 | 0.068 | 0.286 | 0 | 0 | — | — | — |

## Worst-N Outliers

| Library | Case | JD(TT) | Error (mas) |
|---|---|---|---|

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
    "erfa": "GMST=IAU2006 (eraGmst06), ERA=IAU2000 (eraEra00)",
    "siderust": "GST polynomial (IAU 2006 coefficients), ERA from IERS definition",
    "astropy": "GMST=IAU2006, ERA=IAU2000 via bundled ERFA"
  },
  "mode": "common_denominator"
}
```

## Run Metadata

```json
{
  "date": "2026-02-12T17:55:21Z",
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
# Run date: 2026-02-12T17:55:21Z
# OS: Linux 6.17.0-14-generic  CPU: x86_64
# lab SHA: 7328c9f3e9ed
# siderust SHA: 7e2e9213ef41
# erfa SHA: 1d9738bed995

cd /path/to/siderust/lab
git submodule update --init --recursive
bash run.sh  # or: python3 pipeline/orchestrator.py --experiment gmst_era --n 10000 --seed 42
```
