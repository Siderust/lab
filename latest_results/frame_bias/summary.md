# frame_bias — Summary

Timestamp: 2026-03-12_21-43-00

### Frame Bias (ICRS → Mean J2000)

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-------------------|--------------|----------------|
| siderust | 0.00 | 3.07 | 4.35 | 0.00 | 13.6 | 6.9× |
| astropy | 0.00 | 0.03 | 4.35 | 0.00 | 1942.1 | 0.0× |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | siderust |
|------------|---|---|---|
| frame_bias | IAU 2006 frame bias matrix component from eraBp06 | IAU 2006 frame bias via bundled ERFA (erfa.bp06) | IERS 2003 frame bias via frame rotation provide... |

† Measurement below reliability threshold (<10 ns/op or CV >20%); treat as indicative only.


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
    "erfa": "IAU 2006 frame bias matrix component from eraBp06",
    "siderust": "IERS 2003 frame bias via frame rotation provider (ICRS \u2192 EquatorialMeanJ2000)",
    "astropy": "IAU 2006 frame bias via bundled ERFA (erfa.bp06)",
    "libnova": "Not available (no frame bias concept in libnova)",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-parity",
  "accuracy_interpretation": "accuracy vs ERFA reference (IAU frame bias is a fixed rotation)",
  "note": "Frame bias is a small (~17 mas) time-independent rotation between ICRS and mean J2000. libnova has no equivalent \u2014 its results are skipped.",
  "candidate_parity": "model-parity"
}
```
