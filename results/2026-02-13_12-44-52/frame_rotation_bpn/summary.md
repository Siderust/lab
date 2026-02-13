# frame_rotation_bpn — Summary

Timestamp: 2026-02-13_12-44-52

### Frame Rotation (BPN: ICRS → True-of-Date)

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Matrix Frob p50 | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-----------------|-------------------|--------------|----------------|
| siderust | 60.50 | 270.89 | 289.11 | 0.00 | 0.00 | 1995.4 | 80.2× |
| astropy | 0.00 | 2.80 | 3.07 | 0.00 | 0.00 | 89065.3 | 1.8× |
| libnova | 39.43 | 822.59 | 875.90 | — | 0.00 | 32265.7 | 5.0× |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| frame_rotation_bpn | IAU 2006/2000A bias-precession-nutation (eraPnm... | IAU 2006/2000A via bundled ERFA (erfa.pnm06a) | Meeus precession (ζ,z,θ Equ 20.3) + IAU 1980 nu... | IERS 2003 frame bias + Meeus precession (ζ,z,θ)... |


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
    "astropy": "IAU 2006/2000A via bundled ERFA (erfa.pnm06a)",
    "libnova": "Meeus precession (\u03b6,z,\u03b8 Equ 20.3) + IAU 1980 nutation (63-term Table 21A), applied as RA/Dec corrections (no BPN matrix)"
  },
  "mode": "common_denominator",
  "note": "ERFA and Astropy use the same IAU 2006/2000A model (reference). Siderust uses Meeus precession + IAU 1980 nutation (lower fidelity). libnova uses Meeus precession + IAU 1980 nutation via coordinate-level API (no rotation matrix). Differences measure the model gap, not implementation bugs."
}
```
