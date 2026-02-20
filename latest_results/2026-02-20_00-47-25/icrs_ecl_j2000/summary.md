# icrs_ecl_j2000 — Summary

Timestamp: 2026-02-20_00-47-25

### ICRS → Ecliptic J2000

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-------------------|--------------|----------------|
| siderust | 40.89 | 47.81 | 48.01 | 0.00 | 1.8 | 3.9× |
| astropy | 0.00 | 3.07 | 4.35 | 0.00 | 981.7 | 0.0× |
| libnova | 35.84 | 41.58 | 41.58 | 0.00 | 137.4 | 0.1× |
| anise | 35.91 | 41.58 | 41.69 | 0.00 | 1.6 | 4.5× |

### Feature / Model Parity Matrix

| Experiment | erfa | anise | astropy | libnova | siderust |
|------------|---|---|---|---|---|
| icrs_ecl_j2000 | IAU 2006 ecliptic rotation at J2000 epoch via e... | J2000 ↔ ECLIPJ2000 built-in orientation rotatio... | IAU 2006 ecliptic rotation via bundled ERFA (er... | Meeus obliquity (Eq 22.2) applied at J2000 epoch | ICRS → EclipticMeanJ2000 via frame rotation (me... |


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
    "erfa": "IAU 2006 ecliptic rotation at J2000 epoch via eraEcm06",
    "siderust": "ICRS \u2192 EclipticMeanJ2000 via frame rotation (mean obliquity at J2000)",
    "astropy": "IAU 2006 ecliptic rotation via bundled ERFA (erfa.ecm06)",
    "libnova": "Meeus obliquity (Eq 22.2) applied at J2000 epoch",
    "anise": "J2000 \u2194 ECLIPJ2000 built-in orientation rotation (constant obliquity)"
  },
  "model_parity_class": "model-parity",
  "accuracy_interpretation": "accuracy vs ERFA reference (time-independent obliquity)",
  "note": "Time-independent rotation by the mean obliquity at J2000. All IAU-based libraries should agree to \u00b5as level. libnova uses Meeus obliquity which is close but not identical."
}
```
