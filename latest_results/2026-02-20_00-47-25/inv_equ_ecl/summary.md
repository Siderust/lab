# inv_equ_ecl — Summary

Timestamp: 2026-02-20_00-47-25

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| inv_equ_ecl | RBP × ECM06^T composed matrix | rbp × ecm06^T composed via bundled ERFA | ln_get_equ_from_ecl at date (mean-of-date output) | EclipticTrueOfDate → EquatorialMeanOfDate via F... |


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
    "erfa": "RBP \u00d7 ECM06^T composed matrix",
    "siderust": "EclipticTrueOfDate \u2192 EquatorialMeanOfDate via FromEclipticTrueOfDate",
    "astropy": "rbp \u00d7 ecm06^T composed via bundled ERFA",
    "libnova": "ln_get_equ_from_ecl at date (mean-of-date output)",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch"
}
```
