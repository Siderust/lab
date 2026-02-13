# lunar_position — Summary

Date: 2026-02-13

### Moon Geocentric Position

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 2559.3948 | 3992.3824 | 4829.9978 | -46.5073 | 24.2952 |
| astropy | 0.0255 | 0.0588 | 0.0607 | -0.0001 | 0.0003 |
| libnova | 2607.9456 | 5162.0388 | 5484.8921 | 20.2647 | -3.9822 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| lunar_position | Simplified Meeus Ch.47 (major terms only, ~10' ... | Simplified Meeus Ch.47 (same algorithm as ERFA ... | ELP 2000-82B via ln_get_lunar_equ_coords (full ... | ELP 2000 via Moon::get_apparent_topocentric_equ... |


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
  "models": {
    "erfa": "Simplified Meeus Ch.47 (major terms only, ~10' accuracy)",
    "siderust": "ELP 2000 via Moon::get_apparent_topocentric_equ with site at (0,0,0)",
    "astropy": "Simplified Meeus Ch.47 (same algorithm as ERFA adapter)",
    "libnova": "ELP 2000-82B via ln_get_lunar_equ_coords (full model)"
  },
  "note": "ERFA and Astropy use simplified Meeus (~10' accuracy) \u2014 for benchmarking only. Siderust and libnova use full ELP 2000 \u2014 expect ~arcmin-level differences vs reference. No dedicated ERFA Moon ephemeris exists; cross-library comparison is the primary metric."
}
```
