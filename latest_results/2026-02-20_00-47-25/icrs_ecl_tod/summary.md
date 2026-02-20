# icrs_ecl_tod — Summary

Timestamp: 2026-02-20_00-47-25

### ICRS → Ecliptic of Date

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 0.0000 | 0.0031 | 0.0031 | 0.0000 | -0.0000 |
| astropy | 0.0000 | 0.0031 | 0.0043 | 0.0000 | 0.0000 |
| libnova | 1815.1513 | 4731.6894 | 4995.5012 | -2486.9462 | -0.0422 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| icrs_ecl_tod | IAU 2006 equatorial → ecliptic of date via eraE... | IAU 2006 equatorial → ecliptic via bundled ERFA... | Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ | ICRS → ecliptic of date via DirectionAstroExt::... |


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
    "erfa": "IAU 2006 equatorial \u2192 ecliptic of date via eraEqec06",
    "siderust": "ICRS \u2192 ecliptic of date via DirectionAstroExt::to_ecliptic_of_date",
    "astropy": "IAU 2006 equatorial \u2192 ecliptic via bundled ERFA (erfa.eqec06)",
    "libnova": "Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch",
  "accuracy_interpretation": "agreement with ERFA baseline (libnova Meeus obliquity differs)",
  "note": "Similar to equ_ecl but explicitly identified as ICRS \u2192 ecliptic-of-date transform. ERFA/Astropy share IAU 2006 obliquity model. libnova uses Meeus."
}
```
