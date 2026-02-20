# inv_precession_nutation — Summary

Timestamp: 2026-02-20_00-47-25

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| inv_precession_nutation | N×P composed matrix (eraPmat06 × eraNum06a / tr... | N×P composed via bundled ERFA | ln_get_equ_prec + ln_get_equ_nut sequenced / ap... | EquatorialMeanJ2000 ↔ EquatorialTrueOfDate via ... |


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
    "erfa": "N\u00d7P composed matrix (eraPmat06 \u00d7 eraNum06a / transpose)",
    "siderust": "EquatorialMeanJ2000 \u2194 EquatorialTrueOfDate via frame rotation",
    "astropy": "N\u00d7P composed via bundled ERFA",
    "libnova": "ln_get_equ_prec + ln_get_equ_nut sequenced / approximate inverse",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch"
}
```
