# frame_rotation_bpn — Summary

Timestamp: 2026-02-20_00-47-25

### Frame Rotation (BPN: ICRS → True-of-Date)

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Matrix Frob p50 | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-----------------|-------------------|--------------|----------------|
| siderust | 0.00 | 4.35 | 5.32 | 0.00 | 0.00 | 3225.9 | 14.0× |
| astropy | 0.00 | 4.35 | 5.32 | 0.00 | 0.00 | 47574.6 | 1.0× |
| libnova | 34.98 | 104.69 | 1286.98 | — | 0.00 | 12497.0 | 3.6× |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| frame_rotation_bpn | IAU 2006/2000A bias-precession-nutation (eraPnm... | IAU 2006/2000A via bundled ERFA (erfa.pnm06a) | Meeus precession (ζ,z,θ Equ 20.3) + IAU 1980 nu... | IERS 2003 frame bias + IAU 2006 precession + IA... |


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
  "library_notes": {
    "astropy": "The 'astropy' adapter calls ERFA/pyerfa C kernels directly from Python. Accuracy results are identical to ERFA. Performance results measure Python-loop + ERFA-kernel overhead, NOT the high-level astropy.coordinates stack."
  },
  "models": {
    "erfa": "IAU 2006/2000A bias-precession-nutation (eraPnm06a)",
    "siderust": "IERS 2003 frame bias + IAU 2006 precession + IAU 2000B nutation (frame_rotation provider)",
    "astropy": "IAU 2006/2000A via bundled ERFA (erfa.pnm06a)",
    "libnova": "Meeus precession (\u03b6,z,\u03b8 Equ 20.3) + IAU 1980 nutation (63-term Table 21A), applied as RA/Dec corrections (no BPN matrix)",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch",
  "accuracy_interpretation": "agreement with ERFA baseline (models differ across libraries)",
  "mode": "common_denominator",
  "note": "ERFA and Astropy use the same IAU 2006/2000A model (reference). Siderust uses IAU 2006 precession + IAU 2000B nutation (close to ERFA, with 2000B vs 2000A differences). libnova uses Meeus precession + IAU 1980 nutation via coordinate-level API (no rotation matrix). Differences measure the model gap, not implementation bugs."
}
```
