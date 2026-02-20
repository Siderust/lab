# inv_icrs_ecl_j2000 — Summary

Timestamp: 2026-02-20_00-47-25

### Feature / Model Parity Matrix

| Experiment | erfa | anise | astropy | libnova | siderust |
|------------|---|---|---|---|---|
| inv_icrs_ecl_j2000 | Transpose of eraEcm06(J2000) matrix | Transpose of built-in J2000 ↔ ECLIPJ2000 rotation | Transpose of erfa.ecm06(J2000) | ln_get_equ_from_ecl at J2000 (Meeus obliquity) | EclipticMeanJ2000 → ICRS via TransformFrame inv... |


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
    "erfa": "Transpose of eraEcm06(J2000) matrix",
    "siderust": "EclipticMeanJ2000 \u2192 ICRS via TransformFrame inverse",
    "astropy": "Transpose of erfa.ecm06(J2000)",
    "libnova": "ln_get_equ_from_ecl at J2000 (Meeus obliquity)",
    "anise": "Transpose of built-in J2000 \u2194 ECLIPJ2000 rotation"
  },
  "model_parity_class": "model-parity"
}
```
