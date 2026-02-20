# lunar_position — Summary

Timestamp: 2026-02-20_00-47-25

### Moon Geocentric Position

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 0.0000 | 0.0031 | 0.0031 | -0.0000 | 0.0000 |
| astropy | 0.0255 | 0.0588 | 0.0607 | -0.0001 | 0.0003 |
| libnova | 2607.9456 | 5162.0388 | 5484.8921 | 20.2647 | -3.9822 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| lunar_position | Simplified Meeus Ch.47 (major terms only, ~10' ... | Simplified Meeus Ch.47 (same algorithm as ERFA ... | ELP 2000-82B via ln_get_lunar_equ_coords (full ... | Simplified Meeus Ch.47 (major terms only), cent... |


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
  "ephemeris_source": "Meeus/ELP 2000 (varies by library)",
  "library_notes": {
    "astropy": "The 'astropy' adapter calls ERFA/pyerfa C kernels directly from Python. Accuracy results are identical to ERFA. Performance results measure Python-loop + ERFA-kernel overhead, NOT the high-level astropy.coordinates stack."
  },
  "models": {
    "erfa": "Simplified Meeus Ch.47 (major terms only, ~10' accuracy)",
    "siderust": "Simplified Meeus Ch.47 (major terms only), centralized in siderust astro module",
    "astropy": "Simplified Meeus Ch.47 (same algorithm as ERFA adapter)",
    "libnova": "ELP 2000-82B via ln_get_lunar_equ_coords (full model)",
    "anise": "SPK translation MOON_J2000 \u2192 EARTH_J2000 (DE440 ephemeris, geometric state vector)"
  },
  "model_parity_class": "model-mismatch",
  "accuracy_interpretation": "agreement with ERFA baseline (ERFA uses simplified Meeus, libnova uses full ELP 2000 \u2014 arcminute-level differences are expected)",
  "note": "ERFA, Astropy, and Siderust use the same simplified Meeus benchmark model (~10' accuracy). libnova uses full ELP 2000, so arcminute-level differences vs reference are expected. No dedicated ERFA Moon ephemeris exists; cross-library comparison is the primary metric."
}
```
