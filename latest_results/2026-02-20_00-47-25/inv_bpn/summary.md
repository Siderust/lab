# inv_bpn — Summary

Timestamp: 2026-02-20_00-47-25

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | siderust |
|------------|---|---|---|
| inv_bpn | Transpose of IAU 2006 BPN matrix (eraPnm06a → r... | Transpose of IAU 2006 BPN via bundled ERFA | EquatorialTrueOfDate → ICRS via frame rotation ... |


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
    "erfa": "Transpose of IAU 2006 BPN matrix (eraPnm06a \u2192 rnpb^T)",
    "siderust": "EquatorialTrueOfDate \u2192 ICRS via frame rotation inverse",
    "astropy": "Transpose of IAU 2006 BPN via bundled ERFA",
    "libnova": "Not available (no ICRS/frame bias concept)",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch"
}
```
